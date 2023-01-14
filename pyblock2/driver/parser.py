#! /usr/bin/env python

"""
parser for converting stackblock input and libDMET integral to block2.

Author:
    Huanchen Zhai
    Zhi-Hao Cui
"""

from block2 import VectorUInt8
import numpy as np

KNOWN_KEYS = {"nelec", "spin", "hf_occ", "schedule", "maxiter",
              "twodot_to_onedot", "twodot", "onedot", "zerodot", "sweep_tol",
              "orbitals", "integral_tol", "warmup", "nroots", "outputlevel", "prefix",
              "noreorder", "fiedler", "reorder", "gaopt", "nofiedler", "irrep_reorder",
              "num_thrds", "mkl_thrds", "mem", "intmem", "oh", "nonspinadapted",
              "onepdm", "fullrestart", "restart_onepdm", "restart_oh",
              "twopdm", "restart_twopdm", "startM", "maxM", "symmetrize_ints",
              "diag_twopdm", "restart_diag_twopdm",
              "occ", "bias", "cbias", "correlation", "restart_correlation",
              "lowmem_noise", "dm_noise", "conn_centers", "restart_dir", "cutoff",
              "sym", "irrep", "weights", "statespecific", "mps_tags",
              "tran_onepdm", "tran_twopdm", "restart_tran_onepdm",
              "restart_tran_twopdm", "soc", "overlap", "tran_oh",
              "nat_orbs", "nat_km_reorder", "nat_positive_def",
              "restart_tran_oh", "restart_dir_per_sweep", "fp_cps_cutoff", "mps_dir",
              "full_fci_space", "trans_mps_info", "trunc_type", "decomp_type",
              "orbital_rotation", "delta_t", "target_t", "te_type",
              "read_mps_tags", "compression", "random_mps_init", "trans_mps_to_sz",
              "trans_mps_to_singlet_embedding", "trans_mps_from_singlet_embedding",
              "copy_mps", "restart_copy_mps", "sample", "restart_sample", "resolve_twosz",
              "sample_phase", "sample_reference",
              "extrapolation", "cached_contraction", "singlet_embedding", "normalize_mps",
              "dmrgfci", "mrci", "mrcis", "mrcisd", "mrcisdt", "casci", "nevpt2", "nevpt2s",
              "nevpt2sd", "nevpt2-ijrs", "nevpt2-ij", "nevpt2-rs", "nevpt2-ijr",
              "nevpt2-rsi", "nevpt2-ir", "nevpt2-i", "nevpt2-r", "mrrept2", "mrrept2s",
              "mrrept2sd", "mrrept2-ijrs", "mrrept2-ij", "mrrept2-rs", "mrrept2-ijr",
              "mrrept2-rsi", "mrrept2-ir", "mrrept2-i", "mrrept2-r",
              "big_site", "stopt_dmrg", "stopt_compression", "stopt_sampling",
              "model", "k_symmetry", "k_irrep", "k_mod", "init_mps_center", "heisenberg",
              "use_complex", "real_density_matrix", "expt_algo_type", "one_body_parallel_rule",
              "davidson_max_iter", "davidson_soft_max_iter", "linear_soft_max_iter",
              "n_sub_sweeps", "complex_mps", "split_states", "trans_mps_to_complex",
              "use_general_spin", "trans_integral_to_spin_orbital", "store_wfn_spectra",
              "tran_bra_range", "tran_ket_range", "tran_triangular", "use_hybrid_complex",
              "mem_ratio", "min_mpo_mem", "qc_mpo_type", "full_integral", "skip_inact_ext_sites",
              "proj_weights", "proj_mps_tags", "single_prec", "integral_rescale", "check_dav_tol",
              "simple_parallel", "release_integral", "condense_mpo", "svd_eps", "svd_cutoff",
              "conventional_npdm", "threepdm", "fourpdm", "restart_threepdm", "restart_fourpdm",
              "tran_threepdm", "tran_fourpdm", "restart_tran_threepdm", "restart_tran_fourpdm",
              "restart_nevpt2_npdm", "restart_mps_nevpt", "nevpt_state_num", "openmolcas",
              "fock_fourpdm", "restart_fock_fourpdm", "fock_matrix"}

REORDER_KEYS = {"noreorder", "fiedler", "reorder", "gaopt", "nofiedler",
                "irrep_reorder"}

DYN_CORR_KEYS = {"dmrgfci", "mrci", "mrcis", "mrcisd", "mrcisdt", "casci", "mrrept2",
                 "mrrept2s", "mrrept2sd"}

MRPT_KEYS = {"nevpt2", "nevpt2s", "nevpt2sd", "nevpt2-ijrs", "nevpt2-ij", "nevpt2-rs", "nevpt2-ijr",
             "nevpt2-rsi", "nevpt2-ir", "nevpt2-i", "nevpt2-r"}

RESTART_KEYS = {"restart_onepdm", "restart_twopdm", "restart_threepdm", "restart_fourpdm", "restart_oh",
                "restart_correlation", "restart_tran_onepdm", "restart_tran_twopdm", "restart_fock_fourpdm",
                "restart_tran_threepdm", "restart_tran_fourpdm",
                "restart_tran_oh", "restart_tran_oh", "statespecific",
                "restart_copy_mps", "restart_sample", "orbital_rotation",
                "restart_nevpt2_npdm", "restart_mps_nevpt"}

GAOPT_KEYS = {"maxcomm", "maxgen", "maxcell",
              "cloning", "mutation", "elite", "scale", "method"}

COMPLEX_KEYS = {"use_hybrid_complex", "use_complex"}

def parse(fname):
    """
    parse a stackblock input file.

    Args:
        fname: stackblock input config file.

    Returns:
        dic: dictionary of input args.
    """
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    line = lines[0]

    def read_fcidump(fd):
        with open(fd, 'r') as fin:
            lines = fin.readlines()
        ff = " ".join(lines).lower()
        pars = ff.split('/' if '/' in ff else '&end')[0]
        cont_dict = {}
        p_key = None
        for c in ','.join(pars.split()[1:]).split(','):
            if '=' in c or p_key is None:
                p_key, b = c.split('=')
                cont_dict[p_key.strip().lower()] = b.strip()
            elif len(c.strip()) != 0:
                if len(cont_dict[p_key.strip().lower()]) != 0:
                    cont_dict[p_key.strip().lower()] += ',' + c.strip()
                else:
                    cont_dict[p_key.strip().lower()] = c.strip()
        return cont_dict

    # construct input file from FCIDUMP
    if line.strip().lower().startswith('&fci '):
        cont_dict = read_fcidump(fname)
        lines = ["sym d2h", "orbitals %s" % fname,
                 "nelec %s" % cont_dict.get(
                     'nelec', 0), "spin %s" % cont_dict.get('ms2', 0),
                 "irrep %s" % cont_dict.get(
                     'isym', 0), "hf_occ integral", "schedule default",
                 "maxM 500", "maxiter 30"]

    dic = {}
    schedule = []
    site_dependent_bdims = []
    schedule_start = -1
    schedule_end = -1
    for i, line in enumerate(lines):
        if line.strip() != '' and "schedule" == line.strip().split()[0]:
            line_sp = line.strip().split()[1:]
            if len(line_sp) == 1:
                dic["schedule"] = line_sp[0]
            else:
                schedule_start = i
        elif "end" == line.strip():
            schedule_end = i
        elif schedule_start != -1 and schedule_end == -1 \
                and not line.strip().startswith('!') and not line.strip().startswith('#'):
            abcd = line.split()
            if len(abcd) == 4:
                a, b, c, d = abcd
                site_dependent_bdims.append([])
                schedule.append([int(a), int(b), float(c), float(d)])
            else:
                site_dependent_bdims.append([int(x) for x in abcd[1:-2]])
                schedule.append(
                    [int(abcd[0]), max(site_dependent_bdims[-1]), float(abcd[-2]), float(abcd[-1])])
        elif not line.strip().startswith('!') and not line.strip().startswith('#'):
            line_sp = line.split()
            if len(line_sp) != 0:
                if line_sp[0] in dic:
                    raise ValueError("duplicate key (%s)" % line_sp[0])
                dic[line_sp[0]] = " ".join(line_sp[1:])

    if len(schedule) == 0:
        schedule = get_schedule(dic)
        site_dependent_bdims = [[]] * len(schedule)
    
    if "memory," in dic:
        dic["mem"] = dic["memory,"].replace(",", "")
        del dic["memory,"]
    elif "memory" in dic:
        dic["mem"] = dic["memory"].replace(",", "")
        del dic["memory"]
    
    if "nonspinAdapted" in dic:
        dic["nonspinadapted"] = ""
        del dic["nonspinAdapted"]

    tmp = list(zip(*schedule))
    nsweeps = np.diff(tmp[0]).tolist()
    maxiter = int(dic.get("maxiter", 1)) - int(np.sum(nsweeps))
    assert maxiter > 0
    nsweeps.append(maxiter)

    schedule = [[], [], [], []]
    for nswp, M, tol, noise, SM in zip(nsweeps, *tmp[1:], site_dependent_bdims):
        if dic.get("check_dav_tol", "auto") in ["auto", "1"] and "single_prec" in dic:
            tol = max(5E-6, tol)
        schedule[0].extend([M] * nswp)
        schedule[1].extend([tol] * nswp)
        schedule[2].extend([noise] * nswp)
        schedule[3].extend([SM] * nswp)
    dic["schedule"] = schedule

    # stopt extra keywords
    if "stopt_dmrg" in dic:
        dic["onepdm"] = ""
        dic["diag_twopdm"] = ""
        if "copy_mps" not in dic:
            dic["copy_mps"] = "ZKET"
        if "nonspinadapted" not in dic:
            dic["trans_mps_to_sz"] = ""
    if "stopt_compression" in dic:
        dic["compression"] = ""
        if "mps_tags" not in dic:
            dic["mps_tags"] = "BRA"
        if "read_mps_tags" not in dic:
            if "nonspinadapted" in dic:
                dic["read_mps_tags"] = "ZKET"
            else:
                dic["read_mps_tags"] = "KET"
        if "copy_mps" not in dic:
            dic["copy_mps"] = "ZBRA"
        if "nonspinadapted" not in dic:
            dic["trans_mps_to_sz"] = ""
    if "stopt_sampling" in dic:
        if "mps_tags" not in dic:
            if "nonspinadapted" in dic:
                dic["mps_tags"] = "ZKET ZBRA"
            else:
                dic["mps_tags"] = "KET BRA"

    # model extra keywords
    if "model" in dic:
        dic["noreorder"] = ""

    # optional keywords that can be obtained from fcidump
    if "orbitals" in dic and ("nelec" not in dic or "spin" not in dic or "irrep" not in dic
                              or ("k_symmetry" in dic and "k_irrep" not in dic)):
        if open(dic["orbitals"], 'rb').read(4) != b'\x89HDF':
            cont_dict = read_fcidump(dic["orbitals"])
            if "nelec" not in dic and "nelec" in cont_dict:
                dic["nelec"] = cont_dict["nelec"]
            if "spin" not in dic and "ms2" in cont_dict:
                dic["spin"] = cont_dict["ms2"]
            if "irrep" not in dic and "isym" in cont_dict:
                dic["irrep"] = cont_dict["isym"]
            if "k_symmetry" in dic and "k_irrep" not in dic and "kisym" in cont_dict:
                dic["k_irrep"] = cont_dict["kisym"]

    # sanity check
    diff = set(dic.keys()) - KNOWN_KEYS
    if len(diff) != 0:
        raise ValueError("Unrecognized keys (%s)" % diff)
    if "zerodot" in dic and "twodot" in dic:
        raise ValueError("zerodot conflicits with twodot.")
    if "onedot" in dic and ("twodot_to_onedot" in dic or "twodot" in dic):
        raise ValueError("onedot conflicits with twodot_to_onedot/twodot.")
    if "mem" in dic and (not dic["mem"][-1] in ['g', 'G']):
        raise ValueError("memory unit (%s) should be G" % (dic["mem"][-1]))
    if "intmem" in dic and (not dic["intmem"][-1] in ['g', 'G']):
        raise ValueError("memory unit (%s) should be G" % (dic["intmem"][-1]))
    crs = list(set(dic.keys()) & REORDER_KEYS)
    if len(crs) > 1:
        raise ValueError(
            "Reorder keys %s and %s cannot appear simultaneously." % (crs[0], crs[1]))
    crs = list(set(dic.keys()) & DYN_CORR_KEYS)
    if len(crs) > 1:
        raise ValueError(
            "Dynamic correlation keys %s and %s cannot appear simultaneously."
            % (crs[0], crs[1]))
    crs = list(set(dic.keys()) & COMPLEX_KEYS)
    if len(crs) > 1:
        raise ValueError(
            "Complex keys %s and %s cannot appear simultaneously." % (crs[0], crs[1]))

    # restart check
    if "restart_oh" in dic:
        # OH is always fullrestart, and should not do dmrg or pdm
        dic.pop("onepdm", None)
        dic.pop("twopdm", None)
        dic.pop("correlation", None)
    if "diag_twopdm" in dic:
        dic["onepdm"] = ""
        dic["correlation"] = ""
    if "restart_diag_twopdm" in dic:
        dic["restart_onepdm"] = ""
        dic["restart_correlation"] = ""
    if len(set(dic.keys()) & RESTART_KEYS) != 0:
        dic["fullrestart"] = ""
    if len(set(dic.keys()) & MRPT_KEYS) != 0:
        dic["onepdm"] = ""

    return dic


def parse_gaopt(fname):
    """
    parse a stackblock gaopt input file.

    Args:
        fname: stackblock gaopt input config file.

    Returns:
        dic: dictionary of input args.
    """
    with open(fname, 'r') as fin:
        lines = fin.readlines()
    dic = {}
    for line in lines:
        if not line.strip().startswith('!') and not line.strip().startswith('#'):
            line_sp = line.split()
            if len(line_sp) != 0:
                if line_sp[0] in dic:
                    raise ValueError("duplicate key (%s)" % line_sp[0])
                dic[line_sp[0]] = " ".join(line_sp[1:])

    # sanity check
    diff = set(dic.keys()) - GAOPT_KEYS
    if len(diff) != 0:
        raise ValueError("Unrecognized keys (%s)" % diff)

    return dic


def orbital_reorder(fcidump, method='gaopt'):
    """
    Find an optimal ordering of orbitals for DMRG.
    Ref: J. Chem. Phys. 142, 034102 (2015)

    Args:
        fcidump : block2.FCIDUMP object
        method :
            'gaopt <filename>/default' - genetic algorithm, take several seconds
            'fiedler' - very fast, may be slightly worse than 'gaopt'
            'manual <filename>' - manual reorder read from file
            'irrep <pointgroup>' - group same irrep together

    Return a index array "midx":
        reordered_orb_sym = original_orb_sym[midx]
    """
    from block2 import VectorDouble, OrbitalOrdering, VectorUInt16
    n_sites = fcidump.n_sites
    hmat = fcidump.abs_h1e_matrix()
    xmat = fcidump.abs_exchange_matrix()
    kmat = VectorDouble(np.array(hmat) * 1E-7 + np.array(xmat))
    if method.startswith("gaopt "):
        method = method[len("gaopt "):]
        dic = parse_gaopt(
            method) if method != '' and method != 'default' else {}
        assert dic.get("method", "gauss") == "gauss"
        assert float(dic.get("scale", 1.0)) == 1.0
        n_tasks = int(dic.get("maxcomm", 32))
        opts = dict(
            n_generations=int(dic.get("maxgen", 10000)),
            n_configs=int(dic.get("maxcell", n_sites * 2)),
            n_elite=int(dic.get("elite", 8)),
            clone_rate=1.0 - float(dic.get("cloning", 0.9)),
            mutate_rate=float(dic.get("mutation", 0.1))
        )
        midx, mf = None, None
        for _ in range(0, n_tasks):
            idx = OrbitalOrdering.ga_opt(n_sites, kmat, **opts)
            f = OrbitalOrdering.evaluate(n_sites, kmat, idx)
            idx = np.array(idx)
            if mf is None or f < mf:
                midx, mf = idx, f
    elif method == 'fiedler':
        idx = OrbitalOrdering.fiedler(n_sites, kmat)
        midx = np.array(idx)
    elif method.startswith("irrep "):
        method = method[len("irrep "):]
        if method == 'd2h':
            # D2H
            # 0   1   2   3   4   5   6   7   8   (FCIDUMP)
            #     A1g B3u B2u B1g B1u B2g B3g A1u
            # optimal
            #     A1g B1u B3u B2g B2u B3g B1g A1u
            optimal_reorder = [0, 1, 3, 5, 7, 2, 4, 6, 8]
        elif method == 'c2v':
            # C2V
            # 0  1  2  3  4  (FCIDUMP)
            #    A1 B1 B2 A2
            # optimal
            #    A1 B1 B2 A2
            optimal_reorder = [0, 1, 2, 3, 4]
        else:
            optimal_reorder = [0, 1, 3, 5, 7, 2, 4, 6, 8]
        orb_opt = [optimal_reorder[x] for x in np.array(fcidump.orb_sym)]
        idx = np.argsort(orb_opt)
        midx = np.array(idx)
    elif method.startswith("manual "):
        method = method[len("manual "):]
        idx = [int(x) - 1
               for x in open(method, "r").readline().split()]
        f = OrbitalOrdering.evaluate(n_sites, kmat, VectorUInt16(idx))
        idx = np.array(idx)
        midx, mf = idx, f
    else:
        raise ValueError("Unknown reorder method: %s" % method)
    return midx


def read_integral(fints, n_elec, twos, tol=1e-12, isym=1, orb_sym=None, is_sp=False):
    """
    Read libDMET integral h5py file to block2 FCIDUMP object.

    Args:
        fints: h5 file
        n_elec: number of electrons
        twos: spin, nelec_a - nelec_b
        tol: tolerance of numerical zero
        isym: symmetry
        orb_sym: FCIDUMP orbital symm

    Returns:
        fcidump: block2 integral object.    
    """
    from libdmet_solid.system import integral
    from pyscf import ao2mo
    Ham = integral.load(fints)
    h1e = Ham.H1["cd"]
    g2e = Ham.H2["ccdd"]
    e_core = float(Ham.H0)
    n_sites = Ham.norb
    if is_sp:
        from block2.sp import FCIDUMP
    else:
        from block2 import FCIDUMP
    fcidump = FCIDUMP()

    if len(h1e) == 2:  # UHF
        mh1e_a = h1e[0][np.tril_indices(n_sites)]
        mh1e_a[np.abs(mh1e_a) < tol] = 0.0
        mh1e_b = h1e[1][np.tril_indices(n_sites)]
        mh1e_b[np.abs(mh1e_b) < tol] = 0.0
        mh1e = (mh1e_a, mh1e_b)

        g2e_aa = ao2mo.restore(8, g2e[0], n_sites)
        non0_idx = (g2e_aa < tol)
        non0_idx &= (g2e_aa > -tol)
        g2e_aa[non0_idx] = 0.0

        g2e_bb = ao2mo.restore(8, g2e[1], n_sites)
        non0_idx = (g2e_bb < tol)
        non0_idx &= (g2e_bb > -tol)
        g2e_bb[non0_idx] = 0.0

        g2e_ab = ao2mo.restore(4, g2e[2], n_sites)
        non0_idx = (g2e_ab < tol)
        non0_idx &= (g2e_ab > -tol)
        g2e_ab[non0_idx] = 0.0
        non0_idx = None

        mg2e = (g2e_aa, g2e_bb, g2e_ab)
        fcidump.initialize_sz(n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
    elif len(h1e) == 1:  # RHF
        mh1e = h1e[0][np.tril_indices(n_sites)]
        mh1e[np.abs(mh1e) < tol] = 0.0
        mg2e = ao2mo.restore(8, g2e[0], n_sites)
        non0_idx = (mg2e < tol)
        non0_idx &= (mg2e > -tol)
        mg2e[non0_idx] = 0.0
        non0_idx = None
        fcidump.initialize_su2(n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
    else:
        raise ValueError("h1e dimension error, len = %s", len(h1e))

    if orb_sym is None:
        orb_sym = [1] * n_sites
    fcidump.orb_sym = VectorUInt8(orb_sym)
    return fcidump


def format_schedule(sch):
    bdim, tol, noi, sbdim = sch
    if len(bdim) == 0:
        return ""
    lines = []
    for i in range(0, len(bdim)):
        if len(lines) != 0 and [bdim[i], noi[i], tol[i], sbdim[i]] == lines[-1][2:6]:
            lines[-1][1] = i
        else:
            lines.append([i, i, bdim[i], noi[i], tol[i], sbdim[i]])
    rr = []
    for l in lines:
        rr.append("Sweep%4d-%4d : Mmps = %5d Noise = %9.2g DavTol = %9.2g" % tuple(l[:-1]))
        if len(l[-1]) != 0:
            rr.append("--- SDP Mmps = %20s" % l[-1])
    return rr


def get_schedule(dic):
    start_m = int(dic.get("startM", 250))
    max_m = int(dic.get("maxM", 0))
    if max_m <= 0:
        raise ValueError("A positive maxM must be set for default schedule, " +
                         "current value : %d" % max_m)
    elif max_m < start_m:
        raise ValueError(
            "maxM %d cannot be smaller than startM %d" % (max_m, start_m))
    sch_type = dic.get("schedule", "")
    sweep_tol = float(dic.get("sweep_tol", -1))
    if sweep_tol == -1:
        dic["sweep_tol"] = "1E-5"
        sweep_tol = 1E-5
    schedule = []
    if sch_type == "default":
        def_m = [50, 100, 250, 500] + [1000 * x for x in range(1, 11)]
        def_iter = [8] * 5 + [4] * 9
        def_noise = [1E-3] * 3 + [1E-4] * 2 + [5E-5] * 9
        def_tol = [1E-4] * 3 + [1E-5] * 2 + [5E-6] * 9
        if start_m == max_m:
            schedule.append([0, start_m, 1E-5, 1E-4])
            schedule.append([8, start_m, 5E-6, 5E-5])
        elif start_m < def_m[0]:
            def_m.insert(0, start_m)
            for x in [def_iter, def_noise, def_tol]:
                x.insert(0, x[0])
        elif start_m > def_m[-1]:
            while start_m > def_m[-1]:
                def_m.append(def_m[-1] + 1000)
                for x in [def_iter, def_noise, def_tol]:
                    x.append(x[-1])
        else:
            for i in range(1, len(def_m)):
                if start_m < def_m[i]:
                    def_m[i - 1] = start_m
                    break
        isweep = 0
        for i in range(len(def_m)):
            if def_m[i] >= max_m:
                schedule.append([isweep, max_m, def_tol[i], def_noise[i]])
                isweep += def_iter[i]
                break
            elif def_m[i] >= start_m:
                schedule.append([isweep, def_m[i], def_tol[i], def_noise[i]])
                isweep += def_iter[i]
        if "single_prec" in dic:
            schedule.append([schedule[-1][0] + 8, max_m, 5E-6 if sweep_tol == 0 else sweep_tol / 2.0, 0.0])
        else:
            schedule.append([schedule[-1][0] + 8, max_m, 1E-9 if sweep_tol == 0 else sweep_tol / 10, 0.0])
        last_iter = schedule[-1][0]
        if "twodot" not in dic and "onedot" not in dic and "twodot_to_onedot" not in dic:
            dic["twodot_to_onedot"] = str(last_iter + 2)
        max_iter = int(dic.get("maxiter", 0))
        if max_iter <= schedule[-1][0]:
            dic["maxiter"] = str(last_iter + 4)
            max_iter = last_iter + 4
    else:
        raise ValueError("Unrecognized schedule type (%s)" % sch_type)
    return schedule


def test_read_integral():
    # UHF intergal
    ints = read_integral("ints.h5", n_elec=4, twos=0,
                         tol=1e-12, isym=1, orb_sym=None)
    # RHF integral
    ints = read_integral("ints_res.h5", n_elec=16, twos=0,
                         tol=1e-8, isym=1, orb_sym=None)


if __name__ == "__main__":
    dic = parse(fname="./dmrg.conf")
    print(dic)
    test_read_integral()
