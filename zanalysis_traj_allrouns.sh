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

mkdir Unique_Analysis_Traj
cp analysis_*_traj_unique_min.py  Unique_Analysis_Traj
cd Unique_Analysis_Traj

echo 
echo "First Round Simulation Starting at "`date`
python -u  analysis_split_traj_unique_min.py  > result_analysis_round01.out
echo "First Round Simulation Ending at "`date`
echo "Completed First  Round "

for i in {1..10}; do mkdir Second_Set$i; done
mv *_500.traj *_1000.traj *_1500.traj Second_Set1
mv *_2000.traj *_2500.traj Second_Set2
mv *_3000.traj *_3500.traj *_4000.traj  Second_Set3
mv *_4500.traj *_5000.traj Second_Set4
mv *_5500.traj *_6000.traj *_6500.traj Second_Set5
mv *_7000.traj *_7500.traj Second_Set6
mv *_8000.traj *_8500.traj *_9000.traj Second_Set7
mv *_9500.traj *_10000.traj Second_Set8
mv *_10500.traj *_11000.traj *_11500.traj Second_Set9
mv *_12000.traj *_12500.traj Second_Set10


for folder in Second_Set*; do
	cd $folder;
		echo "Second Round Simulation Starting at "`date`
		cp ../analysis_traj_unique_min.py .
		python -u  analysis_traj_unique_min.py  > result_analysis_round02.out 
		mv unique_min_analysis.traj unique_min_analysis_${folder}.traj
		echo "Second Round Simulation Ending at "`date`
	cd ..
done
echo "Completed Second Round "

for i in {1..5}; do mkdir Third_Set$i; done
mv Second_Set1/unique_min_analysis_Second_Set1.traj Second_Set2/unique_min_analysis_Second_Set2.traj Third_Set1
mv Second_Set3/unique_min_analysis_Second_Set3.traj Second_Set4/unique_min_analysis_Second_Set4.traj Third_Set2
mv Second_Set5/unique_min_analysis_Second_Set5.traj Second_Set6/unique_min_analysis_Second_Set6.traj Third_Set3
mv Second_Set7/unique_min_analysis_Second_Set7.traj Second_Set8/unique_min_analysis_Second_Set8.traj Third_Set4
mv Second_Set9/unique_min_analysis_Second_Set9.traj Second_Set10/unique_min_analysis_Second_Set10.traj Third_Set5

for folder in Third_Set*; do
	cd $folder;
		echo "Third Round Simulation Starting at "`date`
		cp ../analysis_traj_unique_min.py .
		python -u  analysis_traj_unique_min.py  > result_analysis_round03.out
		mv unique_min_analysis.traj unique_min_analysis_${folder}.traj
		echo "Third Round Simulation Ending at "`date`
	cd ..
done
echo "Completed Third Round "

mkdir Fourth_Set1 Fourth_Set2
mv Third_Set1/unique_min_analysis_Third_Set1.traj Third_Set2/unique_min_analysis_Third_Set2.traj Third_Set3/unique_min_analysis_Third_Set3.traj Fourth_Set1
mv Third_Set4/unique_min_analysis_Third_Set4.traj Third_Set5/unique_min_analysis_Third_Set5.traj Fourth_Set2

for folder in Fourth_Set*; do
        cd $folder;
                echo "Fourth Round Simulation Starting at "`date`
                cp ../analysis_traj_unique_min.py .
                python -u  analysis_traj_unique_min.py  > result_analysis_round04.out
                mv unique_min_analysis.traj unique_min_analysis_${folder}.traj
                echo "Fourth Round Simulation Ending at "`date`
        cd ..
done
echo "Completed Fourth Round "
 
mkdir Final_Set 
mv Fourth_Set1/unique_min_analysis_Fourth_Set1.traj Fourth_Set2/unique_min_analysis_Fourth_Set2.traj Final_Set

for folder in Final_Set; do
        cd $folder;
                echo "Final Round Simulation Starting at "`date`
                cp ../analysis_traj_unique_min.py .
                python -u  analysis_traj_unique_min.py  > result_analysis_round05.out
                #mv unique_min_analysis.traj unique_min_analysis_${folder}.traj
                echo "Final Round Simulation Ending at "`date`
        cd ..
done
echo "Completed Fifth Round "
cp Final_Set/unique_min_analysis.traj ../

