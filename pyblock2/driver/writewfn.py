#! /usr/bin/env python
"""
Read block2 MPS,
Write StackBlock MPS.

Needs pyblock (https://github.com/hczhai/pyblock).
Check README of block2 if you get `double free` error from python
when import pyblock and block2 in the same script.

Author:
    Huanchen Zhai
    Jun 3, 2021
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
        if sys.argv[i] in ['-su2', '-sz']:
            arg_dic[sys.argv[i][1:]] = ''
        elif sys.argv[i].startswith('-'):
            arg_dic[sys.argv[i][1:]] = sys.argv[i + 1]
        elif i == 1:
            arg_dic['config'] = sys.argv[i]
else:
    raise ValueError("""
        Usage:
            (A) python writewfn.py -config dmrg.conf -out ./out
            (B) python writewfn.py dmrg.conf
            (C) python writewfn.py -integral FCIDUMP -prefix ./scratch -su2
            (D) python writewfn.py -integral FCIDUMP -prefix ./scratch -sz
            (E) python writewfn.py ... -sym c1

        Args:
            config: StackBlock input file
            out: dir for storing StackBlock MPS (not including 'node0')

            when no config file is given/available:
                integral: path to integral file
                prefix: path to StackBlock MPS (does not include 'node0')
                su2/sz: spin-adapted or non-spin-adapted
                sym: point group name (default is d2h)
                    note that d2h also works for c1

            when the prefix is set in config file using relative path,
                it will be considered as relative to the dir of dmrg.conf
                but not relative to the current dir

        If StackBlock is run with reordered integrals, before running this script,
        a reordered FCIDUMP should be generated using
            python gaopt.py -ord RestartReorder.dat -wint FCIDUMP.NEW
        And then use python writewfn.py -integral FCIDUMP.NEW ...

        Notes:
            Only support generating onedot StackBlock Wavefunction
            with canonical form LLL....KR
            (which is the only form supported by StackBlock/OH)
    """)

scratch = '.'
integral = 'FCIDUMP'
dot = 1
su2 = True
sym = "d2h"
load_dir = "./tmp"
mps_tags = ["KET"]
if "config" in arg_dic:
    config = arg_dic["config"]
    dic = parse(config)
    dd = os.path.dirname(config)
    load_dir = dic.get("prefix", "./")
    if not os.path.isabs(load_dir):
        load_dir = ('.' if dd == '' else dd) + "/" + load_dir
    integral = dic["orbitals"]
    if not os.path.isabs(integral):
        integral = ('.' if dd == '' else dd) + "/" + integral
    su2 = "nonspinadapted" not in dic
    mps_tags = dic.get("mps_tags", "KET").split()
    sym = dic.get("sym", "d2h")
if "prefix" in arg_dic:
    load_dir = arg_dic["prefix"]
if "integral" in arg_dic:
    integral = arg_dic["integral"]
if "out" in arg_dic:
    scratch = arg_dic["out"]
if not os.path.exists(scratch + "/node0"):
    os.makedirs(scratch + "/node0")
if "su2" in arg_dic:
    su2 = True
if "sz" in arg_dic:
    su2 = False
if "sym" in arg_dic:
    sym = arg_dic["sym"]

from pyblock.qchem import DMRGDataPage, BlockHamiltonian
from pyblock.qchem.core import BlockSymmetry

from block import VectorInt, VectorMatrix, Matrix, save_rotation_matrix
from block.dmrg import SweepParams
from block.io import Global
from block.symmetry import StateInfo, VectorSpinQuantum, SpinQuantum, SpinSpace, IrrepSpace
from block.symmetry import state_tensor_product_target, state_tensor_product
from block.operator import Wavefunction


from block2 import SU2, SZ, VectorDouble, VectorUInt16
from block2 import init_memory, set_mkl_num_threads, QCTypes
from block2 import PointGroup, VectorUInt8, FCIDUMP, SeqTypes, NoiseTypes

if su2:
    SX = SU2
    from block2.su2 import MPSInfo, MPS, CG, HamiltonianQC, Expect, SparseMatrix
    from block2.su2 import StateInfo as B2StateInfo
    from block2.su2 import MPOQC, RuleQC, SimplifiedMPO, MovingEnvironment, DMRG, OperatorFunctions
else:
    SX = SZ
    from block2.sz import MPSInfo, MPS, CG, HamiltonianQC, Expect, SparseMatrix
    from block2.sz import StateInfo as B2StateInfo
    from block2.sz import MPOQC, RuleQC, SimplifiedMPO, MovingEnvironment, DMRG, OperatorFunctions

# block2 part

assert os.path.isdir(load_dir)

swap_pg = getattr(PointGroup, "swap_" + sym)
npg = {'d2h': 8, 'c2h': 4, 'c2v': 4, 'd2': 4,
       'ci': 2, 'c2': 2, 'cs': 2, 'c1': 1}[sym]
inv_swap_pg = list(np.argsort([swap_pg(x + 1) for x in range(0, npg)]) + 1)

memory = int(10 * 1E9)
init_memory(isize=int(memory * 0.1), dsize=int(memory * 0.9), save_dir=load_dir)
fcidump = FCIDUMP()
fcidump.read(integral)
n_sites = fcidump.n_sites

vaccum = SX(0)
target = SX(fcidump.n_elec, fcidump.twos, swap_pg(fcidump.isym))

orb_sym = VectorUInt8(map(swap_pg, fcidump.orb_sym))
hamil = HamiltonianQC(vaccum, n_sites, orb_sym, fcidump)
hamil.opf.seq.mode = SeqTypes.Simple


if os.path.isfile(load_dir + "/%s-mps_info.bin" % mps_tags[0]):
    print('load MPSInfo from', load_dir + "/%s-mps_info.bin" % mps_tags[0])
    mps_info = MPSInfo(0)
    mps_info.load_data(load_dir + "/%s-mps_info.bin" % mps_tags[0])
else:
    print('create MPSInfo')
    mps_info = MPSInfo(n_sites, vaccum, target, hamil.basis)
    mps_info.tag = mps_tags[0]

mps = MPS(mps_info).deep_copy(mps_info.tag + "-TMP")
mps.info.load_mutable()

max_bdim = max([x.n_states_total for x in mps.info.left_dims])
if mps.info.bond_dim < max_bdim:
    mps.info.bond_dim = max_bdim
max_bdim = max([x.n_states_total for x in mps.info.right_dims])
if mps.info.bond_dim < max_bdim:
    mps.info.bond_dim = max_bdim

mps.load_data()

cg = CG(200)
cg.initialize()
orig_cf = mps.canonical_form
if mps.canonical_form[0] == 'C' and mps.canonical_form[1] == 'R':
    mps.canonical_form = 'K' + mps.canonical_form[1:]
    mps.center = 0
elif mps.canonical_form[-1] == 'C' and mps.canonical_form[-2] == 'L':
    mps.canonical_form = mps.canonical_form[:-1] + 'S'
    mps.center = mps.n_sites - 1
while mps.center > mps.n_sites - 2:
    mps.move_left(cg, None)
while mps.center < mps.n_sites - 2:
    mps.move_right(cg, None)
if mps.canonical_form[mps.center] == 'S':
    mps.flip_fused_form(mps.center, cg, None)
print(orig_cf, '->', mps.canonical_form, mps.center)

center = mps.center

mps.save_data()
mps.load_mutable()

# pyblock part

page = DMRGDataPage(save_dir=scratch)

opts = dict(fcidump=integral, pg=sym,
            su2=su2, output_level=-1, memory=25000,
            omp_threads=1, mkl_threads=1, page=page)

hamil_cxt = BlockHamiltonian.get(**opts)
xhamil = hamil_cxt.__enter__()

site_sts = xhamil.site_state_info
swpp = SweepParams()
Global.dmrginp.load_prefix = scratch + "/node0"
Global.dmrginp.save_prefix = scratch + "/node0"

forward = True

assert center == hamil.n_sites - dot - 1

stls = []
for i in range(0, center + 1):
    qs = VectorSpinQuantum()
    ns = VectorInt()
    sti = mps.info.left_dims[i + 1]
    dx = list(range(sti.n))
    dx.sort(key=lambda j:
        (sti.quanta[j].n, sti.quanta[j].twos, inv_swap_pg[sti.quanta[j].pg]))
    for j in dx:
        qx = sti.quanta[j]
        qq = SpinQuantum(qx.n, SpinSpace(qx.twos), IrrepSpace(inv_swap_pg[qx.pg] - 1))
        qs.append(qq)
        ns.append(sti.n_states[j])
    st = StateInfo(qs, ns)
    if su2:
        st.save(forward, VectorInt(range(i + 1)), 0)
        st.save(forward, VectorInt(range(i + 1)), -1)
    else:
        st.save(forward, VectorInt(range((i + 1) * 2)), 0)
        st.save(forward, VectorInt(range((i + 1) * 2)), -1)
    stls.append(st)


strs = []
for i in range(center + dot, hamil.n_sites):
    qs = VectorSpinQuantum()
    ns = VectorInt()
    sti = mps.info.right_dims[i]
    dx = list(range(sti.n))
    dx.sort(key=lambda j:
        (sti.quanta[j].n, sti.quanta[j].twos, inv_swap_pg[sti.quanta[j].pg]))
    for j in dx:
        qx = sti.quanta[j]
        qq = SpinQuantum(qx.n, SpinSpace(qx.twos), IrrepSpace(inv_swap_pg[qx.pg] - 1))
        qs.append(qq)
        ns.append(sti.n_states[j])
    st = StateInfo(qs, ns)
    if su2:
        st.save(not forward, VectorInt(range(i, hamil.n_sites)), 0)
        st.save(not forward, VectorInt(range(i, hamil.n_sites)), -1)
    else:
        st.save(not forward, VectorInt(range(i * 2, hamil.n_sites * 2)), 0)
        st.save(not forward, VectorInt(range(i * 2, hamil.n_sites * 2)), -1)
    strs.append(st)


def swap_order_left(idx):
    dd = {}
    mps.info.load_left_dims(idx)
    mps.info.load_left_dims(idx + 1)
    l, m, r = mps.info.left_dims[idx], hamil.basis[idx], mps.info.left_dims[idx + 1]
    clm = B2StateInfo.get_connection_info(l, m, r)
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
    mps.info.load_right_dims(idx)
    mps.info.load_right_dims(idx + 1)
    l, m, r = mps.info.right_dims[idx], hamil.basis[idx], mps.info.right_dims[idx + 1]
    clm = B2StateInfo.get_connection_info(m, r, l)
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


for i in range(1, center):
    stl = stls[i - 1]
    str = site_sts[i][0]
    st = state_tensor_product(stl, str)
    st.collect_quanta()
    rot = [None] * len(st.quanta)
    swod = swap_order_left(i)
    for k in range(len(st.quanta)):
        xq = st.quanta[k]
        q = SX(xq.n, xq.s.irrep, swap_pg(xq.symm.irrep + 1))
        iq = mps.tensors[i].info.find_state(q)
        if iq == -1 or iq >= mps.tensors[i].info.n:
            rot[k] = []
            continue
        mat = mps.tensors[i][iq]
        rot[k] = np.zeros((mat.m, mat.n))
        rot[k][swod[q], :] = np.array(mat, copy=True)
    mrot = [None] * len(st.quanta)
    for ir, r in enumerate(rot):
        if len(r) == 0:
            mrot[ir] = Matrix()
        else:
            mrot[ir] = Matrix(np.ascontiguousarray(r))
    mrot = VectorMatrix(mrot)
    if su2:
        save_rotation_matrix(VectorInt(range(i + 1)), mrot, 0)
        save_rotation_matrix(VectorInt(range(i + 1)), mrot, -1)
    else:
        save_rotation_matrix(VectorInt(range(2 * (i + 1))), mrot, 0)
        save_rotation_matrix(VectorInt(range(2 * (i + 1))), mrot, -1)

if su2:
    wave_sites = VectorInt(range(0, center + 1))
else:
    wave_sites = VectorInt(range(0, 2 * (center + 1)))
wll = state_tensor_product(stls[center - 1], xhamil.site_state_info[center][0])
wll.collect_quanta()
wrr = strs[0]
llrr = state_tensor_product_target(wll, wrr)
wave = Wavefunction()
target_state_info = BlockSymmetry.to_state_info([(xhamil.target, 1)])
wave.initialize(target_state_info.quanta, wll, wrr, dot == 1)
wave.clear()

assert wave.onedot == (dot == 1)

wfn = mps.tensors[center]

swl = swap_order_left(center)
swr = swap_order_right(center + 1)
for (il, ir), mat in wave.non_zero_blocks:
    xql = wll.quanta[il]
    xqr = wrr.quanta[ir]
    ql = SX(xql.n, xql.s.irrep, swap_pg(xql.symm.irrep + 1))
    qr = SX(xqr.n, xqr.s.irrep, swap_pg(xqr.symm.irrep + 1))
    q = wfn.info.delta_quantum.combine(ql, -qr)
    iq = wfn.info.find_state(q)
    if iq == -1:
        continue
    xmat = wfn[iq]
    assert xmat.m == mat.ref.shape[0]
    assert xmat.n == mat.ref.shape[1]
    xx = np.array(mat.ref, copy=True)
    xy = np.array(mat.ref, copy=True)
    xx[swl[ql], :] = np.array(xmat, copy=True)
    xy[:, swr[qr]] = np.ascontiguousarray(xx)
    f = -1 if ql.twos == -2 or ql.twos == 2 else 1
    mat.ref[:, :] = f * np.ascontiguousarray(xy)


wave.save_wavefunction_info(llrr, wave_sites, 0)
wave.save_wavefunction_info(llrr, wave_sites, -1)

swpp.save_state(not forward, 0)
