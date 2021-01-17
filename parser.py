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
              "nonspinadapted", "noreorder", "num_thrds", "mem", 
              "onepdm", "fullrestart", "restart_onepdm", "restart_oh",
              "occ", "bias"}

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
        if "schedule" == line.strip():
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
        raise ValueError("Unrecognized keys (%s)" %diff)
    if not "nonspinadapted" in dic:
        raise ValueError("nonspinadapted should be set.")
    if "onedot" in dic:
        raise ValueError("onedot is currently not supported.")
    if "mem" in dic and (not dic["mem"][-1] in ['g', 'G']):
        raise ValueError("memory unit (%s) should be G" % (dic["mem"][-1]))
    return dic

def read_integral(fints, n_elec, twos, tol=1e-13, isym=1, orb_sym=None):
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
                assert abs(xh1e[i, j] - xh1e[j, i]) < tol
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

if __name__ == "__main__":
    dic = parse(fname="./test/dmrg.conf.6")
    print (dic)
