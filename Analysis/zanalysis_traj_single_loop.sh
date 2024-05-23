#!/bin/bash
#SBATCH --job-name=drl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu 8G
#SBATCH --partition=l_long
#SBATCH --qos=ll


cd "$PBS_O_WORKDIR"
export OMP_NUM_THREADS=1

echo "Starting at "`date`
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR


echo "gym_trpo_parallel_training_ver50_expt1.py"

# Source conda.sh to make conda executables available in the environment
source "/ddn/home/raraju042/anaconda3/etc/profile.d/conda.sh"

# Activate Conda environment which you created and tested earlier
conda activate clusgym_dscribe

#cd result_clusgym_ver50_expt1
mkdir Analysis_Unique_Traj
#cp ../analysis_split_traj_unique_min_idxtrajs.py .

echo 
echo "First Round Simulation Starting at "`date`
python -u  analysis_unique_min_single_loop.py  > result_analysis_unique_min_single_loop.out
echo "First Round Simulation Ending at "`date`
echo "Completed First  Round "

