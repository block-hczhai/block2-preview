# block2_solver
Interface to block2 for libDMET

## Usage

1. Normal mode

        python block2_driver.py dmrg.conf

2. Memory saving mode (use two separate calls with the same input file)

        python block2_driver.py dmrg.conf pre
        python block2_driver.py dmrg.conf run

3. Hybrid openMP/MPI mode

        mpirun -n $NPROC python block2_driver.py dmrg.conf
        mpirun --bind-to core --map-by ppr:$NPROC:node:pe=$NTHRDS python block2_driver.py dmrg.conf

        # you can call `pre` with mpirun, but there will be no speed up (`pre` only takes ~1 min):
        mpirun -n $NPROC python block2_driver.py dmrg.conf pre
        mpirun -n $NPROC python block2_driver.py dmrg.conf run

        # you can also call `pre` without mpirun, and then `run` with mpirun:
        python block2_driver.py dmrg.conf pre
        mpirun -n $NPROC python block2_driver.py dmrg.conf run

4. Orbital ordering using genetic algorithm

        python gaopt_driver.py -config gaopt.conf -integral [FCIDUMP|ints.hf] [-wint FCICUMP.NEW] [-w kmat]
        python gaopt_driver.py -s -config gaopt.conf -integral kmat
        mpirun -n $NPROC python gaopt_driver.py ...


## Input file

Input file format is the same as `dmrg.conf` of `StackBlock 1.5`.

### Supported calculation types:

1. Default: DMRG (output: `E_dmrg.npy`)

        <no extra keywords>
2. Restart DMRG (output: `E_dmrg.npy`)

        fullrestart
3. DMRG + 1PDM (output: `E_dmrg.npy` and `1pdm.npy`)

        onepdm
4. Restart DMRG + 1PDM (output: `E_dmrg.npy` and `1pdm.npy`)

        onepdm
        fullrestart
5. Restart 1PDM (output: `1pdm.npy`)

        restart_onepdm
        fullrestart
6. Restart OH (output: `E_oh.npy`)

        restart_oh
        fullrestart
7. DMRG + 1PDM + 1NPC (output: `E_dmrg.npy`, `1pdm.npy` and `1npc.npy`)

        onepdm
        correlation

### Early DMRG stop

To stop a DMRG run gracefully, e.g., in case of non-convergence, 
create a file named `BLOCK_STOP_CALCULATION` with the text `STOP` (in current directory, not scratch directory).
The DMRG run will then stop as it would be converged after the current sweep is over.

### Special keywords

1. only `noreorder` is supported (reorder has no effect).
2. only `nonspinadapted` is supported for `orbitals` with `hdf5` format .
3. `twodot_to_onedot` is supported. The default is `twodot`.
4. `onedot` will be implicitly used if you restart from a `onedot` mps (can be obtained from previous run with `twodot_to_onedot`).
5. 1pdm can run with either `twodot_to_onedot`, `onedot` or `twodot`.
6. `orbitals` can be either `FCIDUMP` (if the filename ends with `FCIDUMP`) or `hdf5` format.
7. currently, `hf_occ`, `nroots`, and `outputlevel` are ignored.
8. if `warmup occ` then keyword `occ` must be set (a space-separated list of fractional occ numbers).
   Otherwise, no matter what `warmup ???` is set, the CheMPS2 type initial FCI is used.
9. if a line in `dmrg.conf` starts with `!`, the line will be ignored.
10. if `restart_dir` is given, after each sweep, the MPS will be backed up in `restart_dir`.
11. if `conn_centers` is given, the parallelism over sites will be used (MPI required, `twodot` only). For example, `conn_centers auto 5` will divide the processors into 5 groups.
12. if `mps_tags` is given (a single string or a list of strings), the MPS in scratch directory with the specific tag/tags will be loaded for restart (for `statespecific`, `restart_onepdm`, etc.) The default MPS tag for input/output is `KET`.
13. the parameters for the quantum number of the MPS, namely `spin`, `isym` and `nelec` can also take multiple numbers. This can also be combined with `nroots > 1`, which will then enable transition density matrix between MPS with different quantum numbers to be calculated (in a single run). This kind of calulation usually needs a larger `nroots` than the `nroots` actually needed, otherwise, some excited states with different quantum number from the ground-state may be missing.
14. if `soc` keyword is in the input (with no associated value), the (normal or transition) one pdm for triplet excitation operators will be calculated (which can be used for spin-orbit coupling calculation). This keyword should be used together with `onepdm`, `tran_onepdm`, `restart_onepdm`, or `restart_tran_onepdm`. Not supported for `nonspinadapted`.
