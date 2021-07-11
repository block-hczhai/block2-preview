#! /usr/bin/env python
"""
Read StackBlock MPS,
Write block2 MPS.

Needs pyblock (https://github.com/hczhai/pyblock).
Check README of block2 if you get `double free` error from python
when import pyblock and block2 in the same script.

Author:
    Huanchen Zhai
    Apr 15, 2021
"""

try:
    from pyblock2.driver.parser import parse
except ImportError:
    from parser import parse

import sys
import os
import numpy as np

if len(sys.argv) > 1:
    arg_dic = {}
    for i in range(1, len(sys.argv)):
        if sys.argv[i] in ['-su2', '-sz', '-expect', '-reduntant']:
            arg_dic[sys.argv[i][1:]] = ''
        elif sys.argv[i].startswith('-'):
            arg_dic[sys.argv[i][1:]] = sys.argv[i + 1]
        elif i == 1:
            arg_dic['config'] = sys.argv[i]
else:
    raise ValueError("""
        Usage:
            (A) python readwfn.py -config dmrg.conf -out ./out
            (B) python readwfn.py dmrg.conf
            (C) python readwfn.py dmrg.conf -expect
            (D) python readwfn.py dmrg.conf -reduntant
            (E) python readwfn.py -integral FCIDUMP -prefix ./scratch -dot 2 -su2
            (F) python readwfn.py -integral FCIDUMP -prefix ./scratch -dot 2 -sz
            (G) python readwfn.py ... -sym c1

        Args:
            config: StackBlock input file
            out: dir for storing block2 MPS
            expect: if given, the energy expectation value of MPS
                is calculated using block2 and printed at the end
            reduntant: if given, the reduntant parameter in the
                StackBlock MPS will be retained.
                Note that removing reduntant parameters do not
                affect quality of MPS.

            when no config file is given/available:
                integral: path to integral file
                prefix: path to StackBlock MPS (does not include 'node0')
                dot: 1 or 2 dot MPS
                su2/sz: spin-adapted or non-spin-adapted
                sym: point group name (default is d2h)
                    note that d2h also works for c1

            when the prefix is set in config file using relative path,
                it will be considered as relative to the dir of dmrg.conf
                but not relative to the current dir

        If StackBlock is run with reordered integrals, before running this script,
        a reordered FCIDUMP should be generated using
            python gaopt.py -ord RestartReorder.dat -wint FCIDUMP.NEW
        And then use python readwfn.py -integral FCIDUMP.NEW ...
    """)

scratch = './node0'
integral = 'FCIDUMP'
dot = 1
su2 = True
sym = "d2h"
nelec = None
spin = None
hf_occ = None
out_dir = "./out"
expect = "expect" in arg_dic
redunt = "reduntant" in arg_dic
mps_tags = ["KET"]
if "config" in arg_dic:
    config = arg_dic["config"]
    dic = parse(config)
    dd = os.path.dirname(config)
    scratch = dic.get("prefix", "./") + "/node0"
    if not os.path.isabs(scratch):
        scratch = ('.' if dd == '' else dd) + "/" + scratch
    integral = dic["orbitals"]
    if not os.path.isabs(integral):
        integral = ('.' if dd == '' else dd) + "/" + integral
    dot = 1 if "twodot_to_onedot" in dic or "onedot" in dic else 2
    su2 = "nonspinadapted" not in dic
    mps_tags = dic.get("mps_tags", "KET").split()
    sym = dic.get("sym", "d2h")
    if "spin" in dic:
        spin = int(dic["spin"])
    if "nelec" in dic:
        nelec = int(dic["nelec"])
    if "hf_occ" in dic:
        hf_occ = dic["hf_occ"]
if "prefix" in arg_dic:
    scratch = arg_dic["prefix"] + "/node0"
if "integral" in arg_dic:
    integral = arg_dic["integral"]
if "out" in arg_dic:
    out_dir = arg_dic["out"]
if os.path.isfile(scratch + "/wave-0-3.0.0.tmp"):
    dot = 1 if open(scratch + "/wave-0-3.0.0.tmp", 'rb').read(41)[-1] else 2
elif os.path.isfile(scratch + "/wave-0-1.0.0.tmp"):
    dot = 1 if open(scratch + "/wave-0-1.0.0.tmp", 'rb').read(41)[-1] else 2
if "su2" in arg_dic:
    su2 = True
if "sz" in arg_dic:
    su2 = False
if "dot" in arg_dic:
    dot = int(arg_dic["dot"])
if "sym" in arg_dic:
    sym = arg_dic["sym"]

from pyblock.qchem import DMRGDataPage, BlockHamiltonian
from pyblock.algorithm import DMRG
from pyblock.qchem.core import BlockSymmetry

from block import VectorInt, VectorMatrix, load_rotation_matrix
from block.dmrg import SweepParams
from block.io import Global
from block.symmetry import StateInfo, state_tensor_product_target, state_tensor_product
from block.operator import Wavefunction

page = DMRGDataPage(save_dir=scratch)

opts = dict(fcidump=integral, pg=sym,
            su2=su2, output_level=-1, memory=25000,
            omp_threads=1, mkl_threads=1, page=page)

if spin is not None:
    opts['spin'] = spin
if nelec is not None:
    opts['nelec'] = nelec
if hf_occ is not None:
    opts['hf_occ'] = hf_occ

hamil_cxt = BlockHamiltonian.get(**opts)
hamil = hamil_cxt.__enter__()

site_sts = hamil.site_state_info
swpp = SweepParams()
Global.dmrginp.load_prefix = scratch
Global.dmrginp.save_prefix = scratch
forward = True
center = hamil.n_sites - dot - 1

stls = []
for i in range(0, center):
    st = StateInfo()
    st.load(forward, VectorInt(range(i + 1)), -1)
    stls.append(st)
strs = []
for i in range(center + dot, hamil.n_sites):
    st = StateInfo()
    st.load(not forward, VectorInt(range(i, hamil.n_sites)), -1)
    strs.append(st)

if su2:
    wave_sites = VectorInt(range(0, center + 1))
else:
    wave_sites = VectorInt(range(0, 2 * (center + 1)))
wll = state_tensor_product(stls[center - 1], hamil.site_state_info[center][0])
wll.collect_quanta()
if dot == 1:
    wrr = strs[0]
else:
    wrr = state_tensor_product(strs[0], hamil.site_state_info[center + 1][0])
    wrr.collect_quanta()
llrr = state_tensor_product_target(wll, wrr)
wave = Wavefunction()
target_state_info = BlockSymmetry.to_state_info([(hamil.target, 1)])
wave.initialize(target_state_info.quanta, wll, wrr, dot == 1)
wave.load_wavefunction_info(llrr, wave_sites, 0, False)
assert wave.onedot == (dot == 1)

rots = []
for i in range(1, center):
    rotation_matrix = VectorMatrix()
    if su2:
        load_rotation_matrix(VectorInt(range(i + 1)), rotation_matrix, -1)
    else:
        load_rotation_matrix(
            VectorInt(range(2 * (i + 1))), rotation_matrix, -1)
    rots.append(rotation_matrix)

from block2 import SU2, SZ, Global, VectorDouble, VectorUInt16
from block2 import init_memory, set_mkl_num_threads, QCTypes
from block2 import PointGroup, VectorUInt8, FCIDUMP, SeqTypes, NoiseTypes

if su2:
    SX = SU2
    from block2.su2 import MPSInfo, MPS, StateInfo, HamiltonianQC, Expect, SparseMatrix
    from block2.su2 import MPOQC, RuleQC, SimplifiedMPO, MovingEnvironment, DMRG, OperatorFunctions
else:
    SX = SZ
    from block2.sz import MPSInfo, MPS, StateInfo, HamiltonianQC, Expect, SparseMatrix
    from block2.sz import MPOQC, RuleQC, SimplifiedMPO, MovingEnvironment, DMRG, OperatorFunctions

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

swap_pg = getattr(PointGroup, "swap_" + sym)
npg = {'d2h': 8, 'c2h': 4, 'c2v': 4, 'd2': 4,
       'ci': 2, 'c2': 2, 'cs': 2, 'c1': 1}[sym]
inv_swap_pg = list(np.argsort([swap_pg(x + 1) for x in range(0, npg)]) + 1)

memory = int(10 * 1E9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=out_dir)
fcidump = FCIDUMP()
fcidump.read(integral)
n_sites = fcidump.n_sites

twos = fcidump.twos if spin is None else spin
nelec = fcidump.n_elec if nelec is None else nelec
vacuum = SX(0)
# singlet embedding
left_vacuum = vacuum if twos == 0 else SX(twos, twos, 0)
target = SX(nelec + twos, 0, swap_pg(fcidump.isym))

orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
hamil = HamiltonianQC(vacuum, n_sites, orb_sym, fcidump)
hamil.opf.seq.mode = SeqTypes.Simple
mps_info = MPSInfo(n_sites, vacuum, target, hamil.basis)
mps_info.tag = mps_tags[0]
if redunt:
    mps_info.set_bond_dimension_full_fci(left_vacuum, vacuum)
else:
    mps_info.set_bond_dimension_fci(left_vacuum, vacuum)

mps_info.left_dims[0] = StateInfo(left_vacuum)
mps_info.save_left_dims(0)
for i in range(0, center + 1):
    st = StateInfo()
    xst = stls[i] if i != center else wll
    n = len(xst.quanta)
    st.allocate(n)
    st.n = n
    for j in range(n):
        st.quanta[j] = SX(xst.quanta[j].n, xst.quanta[j].s.irrep,
                          swap_pg(xst.quanta[j].symm.irrep + 1))
        st.n_states[j] = xst.n_states[j]
        if not redunt and mps_info.left_dims_fci[i + 1].find_state(st.quanta[j]) == -1:
            st.n_states[j] = 0
    st.sort_states()
    if not redunt:
        st.collect()
    mps_info.left_dims[i + 1] = st
    mps_info.save_left_dims(i + 1)

mps_info.right_dims[n_sites] = StateInfo(vacuum)
mps_info.save_right_dims(n_sites)
for i in range(center + 1, n_sites):
    st = StateInfo()
    xst = strs[i - center - dot] if i != center + dot - 1 else wrr
    n = len(xst.quanta)
    st.allocate(n)
    st.n = n
    for j in range(n):
        st.quanta[j] = SX(xst.quanta[j].n, xst.quanta[j].s.irrep,
                          swap_pg(xst.quanta[j].symm.irrep + 1))
        st.n_states[j] = xst.n_states[j]
    st.sort_states()
    mps_info.right_dims[i] = st
    mps_info.save_right_dims(i)

mps_info.save_mutable()

mps = MPS(n_sites, center, dot)
mps.initialize(mps_info)


def swap_order_left(idx):
    dd = {}
    mps_info.load_left_dims(idx)
    mps_info.load_left_dims(idx + 1)
    l, m, r = mps_info.left_dims[idx], hamil.basis[idx], mps_info.left_dims[idx + 1]
    clm = StateInfo.get_connection_info(l, m, r)
    for ik in range(r.n):
        bbed = clm.n if ik == r.n - 1 else clm.n_states[ik + 1]
        dx = []
        g = 0
        for bb in range(clm.n_states[ik], bbed):
            ibba = clm.quanta[bb].data >> 16
            ibbb = clm.quanta[bb].data & 0xFFFF
            nx = l.n_states[ibba] * m.n_states[ibbb]
            dx.append((l.quanta[ibba], m.quanta[ibbb], g, g + nx))
            g += nx
        dx.sort(key=lambda x: (
            x[0].n, x[0].twos, inv_swap_pg[x[0].pg], x[1].n, x[1].twos, inv_swap_pg[x[1].pg]))
        pp = []
        for xx in dx:
            pp += list(range(xx[2], xx[3]))
        dd[r.quanta[ik]] = np.argsort(pp)
    return dd


def swap_order_right(idx):
    dd = {}
    mps_info.load_right_dims(idx)
    mps_info.load_right_dims(idx + 1)
    l, m, r = mps_info.right_dims[idx], hamil.basis[idx], mps_info.right_dims[idx + 1]
    clm = StateInfo.get_connection_info(m, r, l)
    for ik in range(l.n):
        bbed = clm.n if ik == l.n - 1 else clm.n_states[ik + 1]
        dx = []
        g = 0
        for bb in range(clm.n_states[ik], bbed):
            ibba = clm.quanta[bb].data >> 16
            ibbb = clm.quanta[bb].data & 0xFFFF
            nx = m.n_states[ibba] * r.n_states[ibbb]
            dx.append((m.quanta[ibba], r.quanta[ibbb], g, g + nx))
            g += nx
        dx.sort(key=lambda x: (
            x[1].n, x[1].twos, inv_swap_pg[x[1].pg], x[0].n, x[0].twos, inv_swap_pg[x[0].pg]))
        pp = []
        for xx in dx:
            pp += list(range(xx[2], xx[3]))
        dd[l.quanta[ik]] = np.argsort(pp)
    return dd


if su2:
    if twos == 0:
        mps.tensors[0].data = np.array([1.0, 1.0, 1.0])
    else:
        # singlet embedding
        assert mps.tensors[0].info.n == len(mps.tensors[0].data)
        mat = np.zeros_like(np.array(mps.tensors[0].data))
        for ix in range(mps.tensors[0].info.n):
            q = mps.tensors[0].info.quanta[ix]
            mat[ix] = -1 if (twos % 2 and q.twos == twos +
                             1) or (not twos % 2 and q.twos == twos - 1) else 1
        mps.tensors[0].data = mat
    mps.tensors[n_sites - 1].data = np.array([1.0, 1.0, 1.0])
else:
    mps.tensors[0].data = np.array([1.0, 1.0, 1.0, 1.0])
    mps.tensors[n_sites - 1].data = np.array([1.0, 1.0, 1.0, 1.0])

for i in range(1, center):
    stl = stls[i - 1]
    str = site_sts[i][0]
    st = state_tensor_product(stl, str)
    st.collect_quanta()
    assert len(st.quanta) == len(rots[i - 1])
    rot = rots[i - 1]
    mps.tensors[i].data[:] = 0
    swod = swap_order_left(i)
    for k in range(len(rot)):
        if rot[k].cols == 0:
            continue
        xq = st.quanta[k]
        q = SX(xq.n, xq.s.irrep, swap_pg(xq.symm.irrep + 1))
        iq = mps.tensors[i].info.find_state(q)
        if not redunt and not (iq != -1 and iq < mps.tensors[i].info.n):
            continue
        assert iq != -1 and iq < mps.tensors[i].info.n
        mat = mps.tensors[i][iq]
        assert rot[k].ref.size == np.array(mat).size
        assert mat.m == rot[k].ref.shape[0]
        assert mat.n == rot[k].ref.shape[1]
        xx = np.array(rot[k].ref, copy=True)
        mps.tensors[i][iq] = np.ascontiguousarray(xx[swod[q], :])

wfn = mps.tensors[center]
wfn.data[:] = 0

swl = swap_order_left(center)
swr = swap_order_right(center + 1)
for (il, ir), mat in wave.non_zero_blocks:
    xql = wll.quanta[il]
    xqr = wrr.quanta[ir]
    ql = SX(xql.n, xql.s.irrep, swap_pg(xql.symm.irrep + 1))
    qr = SX(xqr.n, xqr.s.irrep, swap_pg(xqr.symm.irrep + 1))
    q = wfn.info.delta_quantum.combine(ql, -qr)
    iq = wfn.info.find_state(q)
    if not redunt and iq == -1:
        continue
    assert iq != -1
    xmat = wfn[iq]
    assert xmat.m == mat.ref.shape[0]
    assert xmat.n == mat.ref.shape[1]
    xx = np.array(mat.ref, copy=True)
    xx = xx[swl[ql], :][:, swr[qr]]
    f = -1 if ql.twos == -2 or ql.twos == 2 else 1
    if (not su2) and dot == 2 and qr.n == 2 and qr.twos == 0 and xmat.n == 4:
        xx[:, 1:3] *= -1
    elif (not su2) and dot == 2 and qr.n == 2 and qr.twos == 0 and qr.pg == orb_sym[-1] ^ orb_sym[-2]:
        xx[:, :] *= -1
    wfn[iq] = f * np.ascontiguousarray(xx)

mps.save_mutable()
mps.save_data()
mps.deallocate()
max_bdim = max([x.n_states_total for x in mps_info.left_dims])
if mps_info.bond_dim < max_bdim:
    mps_info.bond_dim = max_bdim
max_bdim = max([x.n_states_total for x in mps_info.right_dims])
if mps_info.bond_dim < max_bdim:
    mps_info.bond_dim = max_bdim
mps_info.save_data(out_dir + '/mps_info.bin')
mps_info.save_data(out_dir + '/%s-mps_info.bin' % mps_tags[0])

if expect:
    mpo = MPOQC(hamil, QCTypes.Conventional)
    mpo = SimplifiedMPO(mpo, RuleQC(), True)
    me = MovingEnvironment(mpo, mps, mps, "DMRG")
    me.init_environments(False)
    ex = Expect(me, max_bdim, max_bdim).solve(False)
    print(ex)
