#! /usr/bin/env python

"""
parser for converting stackblock input and libDMET integral to block2.

Author:
    Huanchen Zhai
    Zhi-Hao Cui
"""

from block2 import FCIDUMP
from block2 import VectorUInt8
import numpy as np

KNOWN_KEYS = {"nelec", "spin", "hf_occ", "schedule", "maxiter",
              "twodot_to_onedot", "twodot", "onedot", "sweep_tol",
              "orbitals", "warmup", "nroots", "outputlevel", "prefix",
              "noreorder", "fiedler", "reorder", "gaopt", "nofiedler",
              "num_thrds", "mem", "oh", "nonspinadapted",
              "onepdm", "fullrestart", "restart_onepdm", "restart_oh",
              "twopdm", "restart_twopdm", "startM", "maxM",
              "occ", "bias", "cbias", "correlation", "restart_correlation",
              "lowmem_noise", "conn_centers", "restart_dir", "cutoff",
              "sym", "irrep", "weights", "statespecific", "mps_tags",
              "tran_onepdm", "tran_twopdm", "restart_tran_onepdm",
              "restart_tran_twopdm", "soc", "overlap", "tran_oh",
              "restart_tran_oh", "restart_dir_per_sweep", "fp_cps_cutoff",
              "full_fci_space", "trans_mps_info", "trunc_type", "decomp_type"}

GAOPT_KEYS = {"maxcomm", "maxgen", "maxcell",
              "cloning", "mutation", "elite", "scale", "method"}


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
    dic = {}
    schedule = []
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
        elif schedule_start != -1 and schedule_end == -1:
            a, b, c, d = line.split()
            schedule.append([int(a), int(b), float(c), float(d)])
        elif not line.strip().startswith('!'):
            line_sp = line.split()
            if len(line_sp) != 0:
                if line_sp[0] in dic:
                    raise ValueError("duplicate key (%s)" % line_sp[0])
                dic[line_sp[0]] = " ".join(line_sp[1:])

    if len(schedule) == 0:
        schedule = get_schedule(dic)

    tmp = list(zip(*schedule))
    nsweeps = np.diff(tmp[0]).tolist()
    maxiter = int(dic["maxiter"]) - int(np.sum(nsweeps))
    assert maxiter > 0
    nsweeps.append(maxiter)

    schedule = [[], [], []]
    for nswp, M, tol, noise in zip(nsweeps, *tmp[1:]):
        schedule[0].extend([M] * nswp)
        schedule[1].extend([tol] * nswp)
        schedule[2].extend([noise] * nswp)
    dic["schedule"] = schedule

    # sanity check
    diff = set(dic.keys()) - KNOWN_KEYS
    if len(diff) != 0:
        raise ValueError("Unrecognized keys (%s)" % diff)
    if "onedot" in dic and "twodot_to_onedot" in dic:
        raise ValueError("onedot conflicits with twodot_to_onedot.")
    if "mem" in dic and (not dic["mem"][-1] in ['g', 'G']):
        raise ValueError("memory unit (%s) should be G" % (dic["mem"][-1]))
    crs = list(set(dic.keys()) & {"noreorder", "fiedler", "reorder", "gaopt", "nofiedler"})
    if len(crs) > 1:
        raise ValueError("Reorder keys %s and %s cannot appear simultaneously." % (crs[0], crx[1]))

    # restart check
    if "restart_oh" in dic:
        # OH is always fullrestart, and should not do dmrg or pdm
        dic["fullrestart"] = " "
        dic.pop("onepdm", None)
        dic.pop("twopdm", None)
        dic.pop("correlation", None)
    if "restart_onepdm" in dic:
        dic["fullrestart"] = " "
    if "restart_twopdm" in dic:
        dic["fullrestart"] = " "
    if "restart_correlation" in dic:
        dic["fullrestart"] = " "
    if "statespecific" in dic:
        dic["fullrestart"] = " "
    if "restart_tran_onepdm" in dic:
        dic["fullrestart"] = " "
    if "restart_tran_twopdm" in dic:
        dic["fullrestart"] = " "
    if "restart_tran_oh" in dic:
        dic["fullrestart"] = " "

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
        if not line.strip().startswith('!'):
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


def read_integral(fints, n_elec, twos, tol=1e-12, isym=1, orb_sym=None):
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
    fcidump = FCIDUMP()

    mh1e_a = np.zeros((n_sites * (n_sites + 1) // 2))
    mh1e_b = np.zeros((n_sites * (n_sites + 1) // 2))
    mh1e = (mh1e_a, mh1e_b)
    for xmh1e, xh1e in zip(mh1e, h1e):
        k = 0
        for i in range(0, n_sites):
            for j in range(0, i + 1):
                #assert abs(xh1e[i, j] - xh1e[j, i]) < tol
                xmh1e[k] = xh1e[i, j]
                k += 1
        xmh1e[np.abs(xmh1e) < tol] = 0.0

    g2e_aa = ao2mo.restore(8, g2e[0], n_sites)
    g2e_bb = ao2mo.restore(8, g2e[1], n_sites)
    g2e_ab = ao2mo.restore(4, g2e[2], n_sites)

    mg2e = (g2e_aa, g2e_bb, g2e_ab)
    for xmg2e in mg2e:
        xmg2e[np.abs(xmg2e) < tol] = 0.0
    fcidump.initialize_sz(
        n_sites, n_elec, twos, isym, e_core, mh1e, mg2e)
    if orb_sym is None:
        orb_sym = [1] * n_sites
    fcidump.orb_sym = VectorUInt8(orb_sym)
    return fcidump


def format_schedule(sch):
    bdim, tol, noi = sch
    if len(bdim) == 0:
        return ""
    lines = []
    for i in range(1, len(bdim)):
        if len(lines) != 0 and [bdim[i], noi[i], tol[i]] == lines[-1][2:5]:
            lines[-1][1] = i
        else:
            lines.append([i, i, bdim[i], noi[i], tol[i]])
    return ["Sweep%4d-%4d : Mmps = %5d Noise = %9.2g DavTol = %9.2g" % tuple(l) for l in lines]


def get_schedule(dic):
    start_m = int(dic.get("startM", 250))
    max_m = int(dic.get("maxM", 0))
    if max_m <= 0:
        raise ValueError("A positive maxM must be set for default schedule, " + 
            "current value : %d" % max_m)
    elif max_m < start_m:
        raise ValueError("maxM %d cannot be smaller than startM %d" % (max_m, start_m))
    sch_type = dic.get("schedule", "")
    sweep_tol = float(dic.get("sweep_tol", 0))
    if sweep_tol == 0:
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
        schedule.append([schedule[-1][0] + 8, max_m, sweep_tol / 10, 0.0])
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


if __name__ == "__main__":
    dic = parse(fname="./test/dmrg.conf.6")
    print(dic)
