#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end  
#SBATCH --mail-user=dqi@princeton.edu  
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=192000

module load anaconda
module load cudatoolkit/8.0 cudnn/cuda-8.0/6.0
source activate py36
jupyter nbconvert --ExecutePreprocessor.timeout=None --to notebook --execute features_cifar10_distance.ipynb --allow_errors=True 

