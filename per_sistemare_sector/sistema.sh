#!/bin/bash
#SBATCH --job-name=Sistema
#SBATCH --output=out_sistema.dat
#SBATCH --mem=2G 
#SBATCH --time=24:00:00 
#SBATCH --mail-user=alessandro.corso@ist.ac.at 
#SBATCH --mail-type=NONE 
#SBATCH --no-requeue 
#SBATCH --export=NONE 
unset SLURM_EXPORT_ENV 
export OMP_NUM_THREADS=1 
export MAMBA_EXE='/nfs/scistore15/saricgrp/acorso/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='y';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
micromamba activate /nfs/scistore15/saricgrp/acorso/y/envs/ivan_env 
srun python3 sistema_bonds_sector.py