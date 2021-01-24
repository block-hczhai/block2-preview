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
