#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH --gres=gpu:2 -c 1
#SBATCH -o make.out
#SBATCH -e make.err
# module load cuda
make
# ./md5_gpu 923da6e7196b3babf5f95908d145b0bc # abaodog
./md5_gpu 5501462a4c13dd55a6b236ef4550e3e4 # Erica
./md5_gpu 40687c8206d15373954d8b27c6724f62 # Jack