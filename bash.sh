#!/bin/bash
#SBATCH --job-name="ex_inference"
#SBATCH --output=ex_inference.%J.out
#SBATCH --error=ex_inference.%J.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=batch-short
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=3-00:00:00

srun -n 1 singularity exec --writable-tmpfs --nv -B /home/Extrapolation_impSampling:/Extrapolation_impSampling ./modulus_22.09.sif python ./Extrapolation_impSampling/ex_inference.py 
