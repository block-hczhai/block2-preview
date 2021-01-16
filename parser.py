#! /usr/bin/env python

from block2 import FCIDUMP
from block2 import VectorUInt8
import numpy as np
from pyscf import ao2mo
from libdmet_solid.system import integral

def parse(fname):
    fin = open(fname, 'r')
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
        else:
            line_sp = line.split()
            if len(line_sp) != 0:
                dic[line_sp[0]] = " ".join(line_sp[1:])
    fin.close()
    
    tmp = list(zip(*schedule))
    nsweeps = np.diff(tmp[0]).tolist()
    maxiter = int(dic["maxiter"]) - np.sum(nsweeps)
    nsweeps.append(maxiter)
    
    schedule = [[], [], []]
    for nswp, M, tol, noise in zip(nsweeps, *tmp[1:]):
        schedule[0].extend([M] * nswp)
        schedule[1].extend([tol] * nswp)
        schedule[2].extend([noise] * nswp)
    dic["schedule"] = schedule
    return dic

def read_integral(fints, n_elec, twos):
    Ham = integral.load(fints)
    h1e = Ham.H1["cd"]
    g2e = Ham.H2["ccdd"]
    e_core = float(Ham.H0)
    n_sites = Ham.norb
    tol = 1e-13
    isym = 1
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
    fcidump.orb_sym = VectorUInt8([1] * n_sites)
    return fcidump

if __name__ == "__main__":
    dic = parse(fname="dmrg.conf.000")
    print (dic)
            
