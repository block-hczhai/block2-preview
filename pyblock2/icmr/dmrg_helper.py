
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

"""
Writing integrals and input files for StackBlock MRPT.

Part of this script is adapted from
[https://bitbucket.org/sandeep-sharma/icnevpt2/src/master/icpt.py]

Original authors: Sandeep Sharma <sanshar@gmail.com>
                  Qiming Sun <osirpt.sun@gmail.com>

Revised: Huanchen Zhai, Mar 5, 2022

"""

import numpy as np
import os, shutil

def write_response_integrals(e_cas, eris, nelec, dm1, orb_sym, ptype, theory, cwd, tol=1E-13):
    ncore, nocc, ncas = eris.ncore, eris.nocc, eris.ncas
    nwav = len(eris.h1eff) - ncore

    output_format = '%20.16f%4d%4d%4d%4d\n'
    filename = os.path.join(cwd, 'FCIDUMP_%s' % ptype)
    from pyscf import tools

    if theory == 'nevpt2':
        h1e = eris.h1eff.copy()
        h1e[:ncore,:ncore] += np.einsum('abcd,cd', eris.get_chem('ccaa'), dm1)
        h1e[:ncore,:ncore] -= 0.5 * np.einsum('abcd,bc', eris.get_chem('caac'), dm1)
        h1e[nocc:,nocc:] += np.einsum('abcd,cd', eris.get_chem('vvaa'), dm1)
        h1e[nocc:,nocc:] -= 0.5 * np.einsum('abcd,bd', eris.get_chem('vava'), dm1)
        ecore_aaac = 2 * np.trace(h1e[:ncore,:ncore])
        paaa = eris.get_chem('paaa')
        if ptype == 'aaav':
            vaaa = paaa[ncore:]
            with open(filename,'w') as f:
                tools.fcidump.write_head(f, len(vaaa), nelec, orbsym=orb_sym[ncore:])
                for i in range(len(vaaa)):
                    for j in range(ncas):
                        for k in range(ncas):
                            for l in range(k + 1):
                                if abs(vaaa[i, j, k, l]) > tol:
                                    f.write(output_format % (vaaa[i, j, k, l], i + 1, j + 1, k + 1, l + 1))
                tools.fcidump.write_hcore(f, h1e[ncore:,ncore:], len(h1e[ncore:]),
                    float_format='%20.16f', tol=tol)
                f.write(output_format % (-e_cas, 0, 0, 0, 0))

        elif ptype == 'aaac':
            caaa = paaa[:nocc]
            with open(filename,'w') as f:
                tools.fcidump.write_head(f, nocc, nelec + 2 * ncore, orbsym=orb_sym[:nocc])
                for i in range(len(caaa)):
                    for j in range(ncas):
                        for k in range(ncas):
                            for l in range(k + 1):
                                if abs(caaa[i, j, k, l]) > tol:
                                    f.write(output_format % (caaa[i, j, k, l],
                                        i + 1, j + 1 + ncore, k + 1 + ncore, l + 1 + ncore))
                tools.fcidump.write_hcore(f, h1e[:nocc,:nocc], nocc,
                    float_format='%20.16f', tol=tol)
                f.write(output_format % (-e_cas - ecore_aaac, 0, 0, 0, 0))

    elif theory == 'mrrept2':
        if ptype == 'aaav':
            wwaa = np.zeros((nwav, nwav, ncas, ncas), dtype=eris.h1eff.dtype)
            wwaa[:ncas, :ncas] = eris.get_chem('aaaa')
            wwaa[ncas:, :ncas] = eris.get_chem('vaaa')
            wwaa[ncas:, ncas:] = eris.get_chem('vvaa')
            vava = eris.get_chem('vava')
            with open(filename,'w') as f:
                tools.fcidump.write_head(f, len(wwaa), nelec, orbsym=orb_sym[ncore:])
                for i in range(len(wwaa)):
                    for j in range(i + 1):
                        for k in range(ncas):
                            for l in range(k + 1):
                                if abs(wwaa[i, j, k, l]) > tol:
                                    f.write(output_format % (wwaa[i, j, k, l], i + 1, j + 1, k + 1, l + 1))
                                if j >= ncas and abs(vava[i - ncas, k, j - ncas, l]) > tol:
                                    f.write(output_format % (vava[i - ncas, k, j - ncas, l], i + 1, k + 1, j + 1, l + 1))
                                if j >= ncas and l != k and abs(vava[i - ncas, l, j - ncas, k]) > tol:
                                    f.write(output_format % (vava[i - ncas, l, j - ncas, k], i + 1, l + 1, j + 1, k + 1))
                tools.fcidump.write_hcore(f, eris.h1eff[ncore:,ncore:], len(eris.h1eff[ncore:]),
                    float_format='%20.16f', tol=tol)
                f.write(output_format % (-e_cas, 0, 0, 0, 0))

        elif ptype == 'aaac':
            oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=eris.h1eff.dtype)
            kmap = {'c': slice(ncore), 'a': slice(ncore, None)}
            for ki, vi in kmap.items():
                for kj, vj in kmap.items():
                    for kk, vk in kmap.items():
                        for kl, vl in kmap.items():
                            part = eris.get_chem(ki + kj + kk + kl)
                            assert part is not None
                            oooo[vi, vj, vk, vl] = part
            hoo = np.zeros((nocc, nocc), dtype=eris.h1eff.dtype)
            for ki, vi in kmap.items():
                for kj, vj in kmap.items():
                    hoo[vi, vj] = getattr(eris, 'h' + ki + kj)
            with open(filename,'w') as f:
                tools.fcidump.write_head(f, nocc, nelec + 2 * ncore, orbsym=orb_sym[:nocc])
                ij = 0
                for i in range(nocc):
                    for j in range(i + 1):
                        kl = 0
                        for k in range(nocc):
                            for l in range(k + 1):
                                if ij >= kl and abs(oooo[i, j, k, l]) > tol:
                                    f.write(output_format % (oooo[i, j, k, l], i + 1, j + 1, k + 1, l + 1))
                                kl += 1
                        ij += 1
                tools.fcidump.write_hcore(f, hoo, nocc, float_format='%20.16f', tol=tol)
                cccc = oooo[:ncore, :ncore, :ncore, :ncore]
                ecr = 2 * np.einsum('iijj', cccc) - np.einsum('ijij', cccc)
                ecx = 2 * np.trace(hoo[:ncore,:ncore])
                f.write(output_format % (-ecr - e_cas - ecx, 0, 0, 0, 0))

def write_dmrg_response_conf_file(mc, ptype='aaac', maxM=0, root=None):
    dmrgci = mc.fcisolver
    cwd = dmrgci.runtimeDir
    nelec = mc.nelecas[0] + mc.nelecas[1]
    spin = mc.nelecas[0] - mc.nelecas[1]
    norbs = mc.mo_coeff.shape[1]

    xopen, xclosed = [], []
    reorder = np.loadtxt("%s/node0/RestartReorder.dat" % (dmrgci.scratchDirectory))
    if ptype == 'aaac':
        xclosed = tuple(range(1, mc.ncore + 1))
        reord = [*[r + 1 + mc.ncore for r in reorder], *xclosed]
    else:
        xopen = tuple(range(1 + mc.ncas, norbs - mc.ncore + 1))
        reord = [*[r + 1 for r in reorder], *xopen]
    with open(os.path.join(cwd, "reorder_%s.txt" % ptype), "w") as f:
        f.write("%d " * len(reord) % tuple(reord) + "\n")
    assert not isinstance(dmrgci.wfnsym, str)
    conf = {
        "nelec": nelec + 2 * mc.ncore * (ptype == 'aaac'),
        "spin": spin,
        "irrep": dmrgci.wfnsym,
        "orbitals": "FCIDUMP_%s" % ptype,
        "sweep_tol": dmrgci.tol,
        "outputlevel": dmrgci.outputlevel,
        "hf_occ": "integral",
        "num_thrds": dmrgci.num_thrds,
        "occ": 10000,
        "reorder": "reorder_%s.txt" % ptype,
        "response%s" % ptype: "",
        "baseStates": 0 if root is None else root,
        "projectorStates": 0 if root is None else root,
        "targetState": 200,
        "partialsweep": mc.ncas,
        "open": "%d " * len(xopen) % xopen,
        "closed": "%d " * len(xclosed) % xclosed,
        "twodot": ""
    }
    if dmrgci.memory is not None:
        conf["memory"] = "%d g" % dmrgci.memory
    if dmrgci.scratchDirectory is not None:
        conf["prefix"] = dmrgci.scratchDirectory
    if dmrgci.groupname is not None:
        from pyscf.dmrgscf import dmrg_sym
        conf["sym"] = dmrg_sym.d2h_subgroup(dmrgci.groupname).lower()
    maxM = max(maxM, dmrgci.maxM)
    with open(os.path.join(cwd, "response_%s.conf" % ptype), "w") as f:
        f.write('schedule\n')
        iter = 0
        for m in range(dmrgci.maxM, maxM + 1, 1000):
            f.write('%6d %6d %8.4e %8.4e\n' % (iter * 4, m, 1E-6, 1E-5))
            iter += 1
        f.write('%6d %6d %8.4e %8.4e\n' % (iter * 4, maxM, 1E-6, 1E-5))
        f.write('end\n')
        conf["maxiter"] = 4 * iter + 4
        for k, v in conf.items():
            f.write("%s %s\n" % (k, v))

def dmrg_response_singles(mc, eris, dm1, ptype, theory, root=None):
    nelec = mc.nelecas[0] + mc.nelecas[1]
    mol = mc._scf.mol
    n_mo = mc.mo_coeff.shape[1]
    if not mol.symmetry:
        orb_sym = [1] * n_mo
    else:
        from pyscf import symm, tools
        orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mc.mo_coeff)
        orbsym = [symm.irrep_name2id(mol.groupname, i) for i in orb_sym_str]
        orb_sym = [tools.fcidump.ORBSYM_MAP[mol.groupname][i] for i in orbsym]
    ecas = mc.e_cas
    if root is not None and isinstance(ecas, np.ndarray):
        ecas = ecas[root]
    write_response_integrals(ecas, eris, nelec, dm1, orb_sym,
        ptype=ptype, theory=theory, cwd=mc.fcisolver.runtimeDir)
    dmrgci = mc.fcisolver
    shutil.copy("%s/node0/RestartReorder.dat" % dmrgci.scratchDirectory,
        "%s/node0/CasReorder.dat" % dmrgci.scratchDirectory)
    from subprocess import check_call
    import struct
    if mc.ncore == 0 and ptype == 'aaac':
        ener = 0.0
    else:
        write_dmrg_response_conf_file(mc, ptype, root=root)
        try:
            output = check_call("%s %s response_%s.conf > response_%s.out 2>&1" %
                (dmrgci.mpiprefix, dmrgci.executable, ptype, ptype),
                cwd=mc.fcisolver.runtimeDir, shell=True)
            with open("%s/node0/dmrg.e" % dmrgci.scratchDirectory, "rb") as f:
                ener = struct.unpack('d', f.read(8))[0]
        except ValueError:
            print(output)
    shutil.copy("%s/node0/CasReorder.dat" % dmrgci.scratchDirectory,
        "%s/node0/RestartReorder.dat" % dmrgci.scratchDirectory)
    return ener
