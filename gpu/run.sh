#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH --gres=gpu:4 -c 1
#SBATCH -o run.out
#SBATCH -e run.err

HASH_TO_CRACK=40687c8206d15373954d8b27c6724f62 # Jack

module load cuda
make

./md5_gpu $HASH_TO_CRACK