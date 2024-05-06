#!/bin/bash
#SBATCH --job-name=drl
#SBATCH --ntasks=24
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


echo "gym_trpo_parallel_training_ver99_expt1.py"

# Source conda.sh to make conda executables available in the environment
source "/ddn/home/raraju042/anaconda3/etc/profile.d/conda.sh"

# Activate Conda environment which you created and tested earlier
conda activate clusgym_dscribe

#cd result_clusgym_ver99_expt1
#cp ../analysis_split_traj_unique_min_idxtrajs.py .

cp ~/bin/DRL_Clus/analysis_unique_traj_multiprocess.py .
cp -r episode_min-COPY round_0
tar -cJvf episode_min.txz episode_min-COPY 
echo
echo
tar -cJvf history.txz history
echo 
echo
tar -cJvf trajs.txz trajs
echo
echo
echo "Started deletion folders after compression"
rm -r episode_min-COPY history plots trajs unique_min unique_min_ene.txt episode_min  
echo "completed deletion folders after compression"
echo
echo
echo 
echo "Mutiprocessing unique multitraj analysis starting at "`date`
python -u  analysis_unique_traj_multiprocess.py
echo "Mutiprocessing unique multitraj analysis ending at "`date`

