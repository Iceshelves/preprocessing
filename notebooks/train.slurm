#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
srun python ~/preprocessing/notebooks/feed_loop_merged.py

