#!/bin/bash
#SBATCH -t 119:59:00
#SBATCH -p gpu
srun python feed_loop_to_merge.py

