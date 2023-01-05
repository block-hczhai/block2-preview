
#  block2: Efficient MPO implementation of quantum chemistry DMRG
#  Copyright (C) 2020-2021 Huanchen Zhai <hczhai@caltech.edu>
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
Transformation of ERIs for multi-reference theories.
"""

from block2 import WickPermutation
import numpy as np

def get_chem_eri(chem_eris, ckey):
    known = chem_eris.known
    perm_chs = WickPermutation.qc_chem()
    perm_chs = WickPermutation.complete_set(4, perm_chs)
    sxx = {
        'c': slice(chem_eris.ncore),
        'a': slice(chem_eris.ncore, chem_eris.nocc),
        'v': slice(chem_eris.nocc, None)
    }
    for kkey in known:
        for perm in perm_chs:
            pp = np.array(perm.data, dtype=int)
            mkey = np.array(list(kkey))[pp]
            if all([p == m or m == 'p' for p, m in zip(ckey, mkey)]):
                sl = [slice(None) if p == m else sxx[p] for p, m in zip(ckey, mkey)]
                return getattr(chem_eris, kkey).transpose(*pp)[tuple(sl)]

def get_phys_eri(chem_eris, pkey):
    kmap = {'A': 'a', 'E': 'v', 'I': 'c'}
    ckey = np.array([kmap.get(k, k) for k in pkey])[np.array([0, 2, 1, 3], dtype=int)]
    return get_chem_eri(chem_eris, ckey).transpose(0, 2, 1, 3)

def get_h1_eri(chem_eris, pkey):
    kmap = {'A': 'a', 'E': 'v', 'I': 'c'}
    return getattr(chem_eris, 'h' + ''.join([kmap.get(k, k) for k in pkey]))

def get_h1eff_eri(chem_eris, pkey):
    kmap = {'A': 'a', 'E': 'v', 'I': 'c'}
    return getattr(chem_eris, 'heff' + ''.join([kmap.get(k, k) for k in pkey]))

class _ChemistsERIs:
    '''(pq|rs)'''
    def __init__(self, mol=None):
        self.mol = mol
    
    def _common_init_(self, mc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mc.mo_coeff
        self.ncore = mc.ncore
        self.ncas = mc.ncas
        self.nocc = self.ncore + self.ncas
        ncore = mc.ncore
        dmcore = np.dot(mo_coeff[:,:ncore], mo_coeff[:,:ncore].T)
        vj, vk = mc._scf.get_jk(mc.mol, dmcore)
        vhfcore = mo_coeff.conj().T @ (vj * 2 - vk) @ mo_coeff
        self.h1eff = mo_coeff.conj().T @ mc.get_hcore() @ mo_coeff + vhfcore
    
    get_chem = get_chem_eri
    get_phys = get_phys_eri
    get_h1 = get_h1_eri
    get_h1eff = get_h1eff_eri

def init_eris(mc, mo_coeff=None, mrci=False):
    from pyscf import ao2mo, lib
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    nmo, ncore, ncas = mo_coeff.shape[1], mc.ncore, mc.ncas
    nocc = ncore + ncas
    nvir = nmo - nocc
    eris = _ChemistsERIs()
    eris._common_init_(mc, mo_coeff)
    if mrci:
        h1e = mo_coeff.conj().T @ mc._scf.get_hcore() @ mo_coeff
        g2e = ao2mo.restore(1, ao2mo.kernel(mc._scf.mol, mo_coeff), nmo)
        eris.known = ['vvca', 'aaca', 'cvca', 'cvaa', 'cvcv', 'caac', 'avaa',
                      'vacc', 'aacc', 'avva', 'avcv', 'accc', 'vaca', 'vvaa',
                      'aaaa', 'vvvv', 'vccc', 'vvva', 'vvvc', 'cccc', 'vvcc']
        imap = {"v" : slice(nocc, None) , "a" : slice(ncore, nocc), "c" : slice(ncore)}
        for k in eris.known:
            setattr(eris, k, g2e[tuple(imap[x] for x in k)].copy())
        for k in [a + b for a in 'cav' for b in 'cav']:
            setattr(eris, 'h' + k, h1e[tuple(imap[x] for x in k)].copy())
        for k in [a + b for a in 'cav' for b in 'cav']:
            setattr(eris, 'heff' + k, eris.h1eff[tuple(imap[x] for x in k)].copy())
    else:
        # adapted from pyscf.mrpt.nevpt2.py _trans
        eris.known = ['ppaa', 'papa', 'pacv', 'cvcv']
        nwav = nmo - ncore # w = a + v
        if mc._scf._eri is None:
            mc._scf._eri = mc._scf.mol.intor('int2e', aosym='s8')
        pwxx = ao2mo.incore.half_e1(mc._scf._eri,
            (mo_coeff[:, :nocc], mo_coeff[:, ncore:]), compact=False)
        pwxx = pwxx.reshape((nocc, nwav, -1))
        cvcv = np.zeros((ncore * nvir, ncore * nvir))
        pacv = np.empty((nmo, ncas, ncore * nvir))
        aapp = np.empty((ncas, ncas, nmo * nmo))
        papa = np.empty((nmo, ncas, nmo * ncas))
        wcv = np.empty((nwav, ncore * nvir))
        wpa = np.empty((nwav, nmo * ncas))
        klcv = (0, ncore, nocc, nmo)
        klpa = (0, nmo, ncore, nocc)
        klpp = (0, nmo, 0, nmo)
        for i, wxx in enumerate(pwxx[:ncore]):
            ao2mo._ao2mo.nr_e2(wxx, mo_coeff, klcv, aosym='s4', out=wcv)
            ao2mo._ao2mo.nr_e2(wxx[:ncas], mo_coeff, klpa, aosym='s4', out=papa[i])
            cvcv[i * nvir:(i + 1) * nvir] = wcv[ncas:]
            pacv[i] = wcv[:ncas]
        for i, wxx in enumerate(pwxx[ncore:nocc]):
            ao2mo._ao2mo.nr_e2(wxx, mo_coeff, klcv, aosym='s4', out=wcv)
            ao2mo._ao2mo.nr_e2(wxx, mo_coeff, klpa, aosym='s4', out=wpa)
            ao2mo._ao2mo.nr_e2(wxx[:ncas], mo_coeff, klpp, aosym='s4', out=aapp[i])
            pacv[ncore:, i] = wcv
            papa[ncore:, i] = wpa
        ppaa = lib.transpose(aapp.reshape(ncas ** 2, -1))
        eris.ppaa = ppaa.reshape(nmo, nmo, ncas, ncas)
        eris.papa = papa.reshape(nmo, ncas, nmo, ncas)
        eris.pacv = pacv.reshape(nmo, ncas, ncore, nvir)
        eris.cvcv = cvcv.reshape(ncore, nvir, ncore, nvir)
    return eris

def init_pdms(mc, pdm_eqs, root=None):
    from pyscf import fci
    xci = mc.ci
    if root is not None and (isinstance(xci, list) or isinstance(xci, range)):
        xci = xci[root]
    if isinstance(xci, list):
        trans_dms = [None] * len(xci)
        for ibi in range(len(xci)):
            dmm = [None] * len(xci)
            for iki in range(len(xci)):
                dms = fci.rdm.make_dm1234('FCI4pdm_kern_sf', xci[ibi], xci[iki],
                    mc.ncas, mc.nelecas)
                dm_names = ["dm1AA", "dm2AAAA", "dm3AAAAAA", "dm4AAAAAAAA"]
                E1, E2, E3, E4 = [np.zeros_like(dm) for dm in dms]
                exec("".join(pdm_eqs), globals(), {
                    "E1": E1, "E2": E2, "E3": E3, "E4": E4, **dict(zip(dm_names, dms)),
                    "deltaAA": np.eye(mc.ncas)
                })
                dmm[iki] = E1, E2, E3, E4
            trans_dms[ibi] = dmm
        return trans_dms
    elif isinstance(xci, int):
        # dmrg solver
        if mc.fcisolver.has_threepdm == False:
            from subprocess import call
            call("rm ./node0/Rotation-*.state-1.tmp >> rm.out 2>&1", cwd=mc.fcisolver.scratchDirectory, shell=True)
            call("rm ./node0/wave-*.-1.tmp >> rm.out 2>&1", cwd=mc.fcisolver.scratchDirectory, shell=True)
            call("rm ./node0/CasReorder.dat >> rm.out 2>&1", cwd=mc.fcisolver.scratchDirectory, shell=True)
        dms = mc.fcisolver._make_dm123(xci, mc.ncas, mc.nelecas)
        dm_names = ["dm1AA", "dm2AAAA", "dm3AAAAAA"]
        E1, E2, E3 = [np.zeros_like(dm) for dm in dms]
        exec("".join(pdm_eqs[:3]), globals(), {
            "E1": E1, "E2": E2, "E3": E3, **dict(zip(dm_names, dms)),
            "deltaAA": np.eye(mc.ncas)
        })
        E4 = None
        if hasattr(mc.fcisolver, 'executable') and mc.fcisolver.executable.strip().endswith('block2main'):
            E4 = mc.fcisolver.make_rdm4(xci, mc.ncas, mc.nelecas, restart=True)
        return E1, E2, E3, E4
    else:
        dms = fci.rdm.make_dm1234('FCI4pdm_kern_sf', xci, xci, mc.ncas, mc.nelecas)
        dm_names = ["dm1AA", "dm2AAAA", "dm3AAAAAA", "dm4AAAAAAAA"]
        E1, E2, E3, E4 = [np.zeros_like(dm) for dm in dms]
        exec("".join(pdm_eqs), globals(), {
            "E1": E1, "E2": E2, "E3": E3, "E4": E4, **dict(zip(dm_names, dms)),
            "deltaAA": np.eye(mc.ncas)
        })
        return E1, E2, E3, E4
