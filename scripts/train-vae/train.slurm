#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
srun python ~/preprocessing/scripts/train-vae/main.py ~/preprocessing/scripts/train-vae/train-vae.ini

