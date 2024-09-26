import os
import sys

MaxRunTime = 240
runfilename = 'analysisfile.sh'
JobName = "Ivan_analysis"
OutFileName = 'out_analysis.dat'
f = open(runfilename, "w")
f.write("""#!/bin/bash
#SBATCH --job-name={}
#SBATCH --output={}
""".format(JobName, OutFileName))
f.write("#SBATCH --mem=2G \n")
f.write("#SBATCH --time={:d}:00:00 \n".format(MaxRunTime))
f.write("#SBATCH --mail-user=alessandro.corso@ist.ac.at \n")
f.write("#SBATCH --mail-type=NONE \n")
f.write("#SBATCH --no-requeue \n")
f.write("#SBATCH --export=NONE \n")
f.write("unset SLURM_EXPORT_ENV \n")
f.write("export OMP_NUM_THREADS=1 \n")

f.write("""export MAMBA_EXE='/nfs/scistore15/saricgrp/acorso/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='y';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
micromamba activate /nfs/scistore15/saricgrp/acorso/y/envs/ivan_env \n""")

f.write('srun python3 analysis.py')
f.close()

#run the analisis

r = os.system("sbatch "+runfilename)