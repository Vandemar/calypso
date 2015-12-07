#!/bin/bash

#SBATCH -J mhdCalypso                  # Job name
#SBATCH -o 4Poster.out              # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu                  # Queue name
#SBATCH -N 1                          # Total number of nodes requested (20 cores/node)
#SBATCH -n 1                         # Total number of mpi tasks requested
#SBATCH -t 00:30:00                   # Run time (hh:mm:ss) - 4 hours

#--------------------------------------------------------------------------
# ---- You normally should not need to edit anything below this point -----
#--------------------------------------------------------------------------

export OMP_NUM_THREADS=16

./heartRate.sh

module load cuda/6.5

#./startRuns control_MHD_template control_sph_shell_template 31
#../../buildOptimal/bin/gen_sph_grids
#ibrun -n 1 -o 0 ../../buildOptimal/bin/sph_mhd
#./startRuns control_MHD_template control_sph_shell_template 63
#../../buildOptimal/bin/gen_sph_grids
#ibrun -n 1 -o 0 ../../buildOptimal/bin/sph_mhd
#./startRuns control_MHD_template control_sph_shell_template 83
#../../buildOptimal/bin/gen_sph_grids
#ibrun -n 1 -o 0 ../../buildOptimal/bin/sph_mhd
#./startRuns control_MHD_template control_sph_shell_template 103
#../../buildOptimal/bin/gen_sph_grids
#ibrun -n 1 -o 0 ../../buildOptimal/bin/sph_mhd
#./startRuns control_MHD_template control_sph_shell_template 127
#../../buildOptimal/bin/gen_sph_grids
ibrun -n 1 -o 0 ../../buildOptimal/bin/sph_mhd

