#!/bin/bash -l
#SBATCH --job-name=drlclus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu 8G


module purge
module load bluebear

cd "$PBS_O_WORKDIR"
export OMP_NUM_THREADS=1

echo "Starting at "`date`
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR


# Source conda.sh to make conda executables available in the environment
source /rds/homes/r/rajur/mambaforge-pypy3/etc/profile.d/conda.sh
conda activate clusgym_dscribe

cd result_clusgym_ver50_expt1
pwd


echo "Starting at "`date`

#tar -cJvf history.txz history
#rm -r history
#echo
#echo

#tar -cJvf unique_min.txz unique_min
rm -r unique_min
echo 
echo 

tar -cJvf episode_min.txz episode_min
echo 
echo 

tar -cJvf trajs.txz trajs
echo
echo

cp ../analysis_unique_min_single_loop.py .


echo "Starting analysis unique trajs at "`date`
python -u analysis_unique_min_single_loop.py > result_analysis_unique_min_single_loop.out
echo "Ending analysis unique trajs  at "`date`


