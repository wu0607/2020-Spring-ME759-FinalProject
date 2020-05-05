#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH --gres=gpu:2 -c 1
#SBATCH -o make.out
#SBATCH -e make.err
# module load cuda
rm md5_gpu
make
./md5_gpu 40687c8206d15373954d8b27c6724f62 # Jack
./md5_gpu 5501462a4c13dd55a6b236ef4550e3e4 # Erica
./md5_gpu 2a21e2561359ebf2fb2d634ee7837a8e # Nvidia
./md5_gpu 4ca4f434da0ea97ebff27833d69728d3 # weekend 224 using 512 thread, 150 using 768 thread
./md5_gpu b49fe33a2152c077d827c0d3db7e68a7 # jaychou
# ./md5_gpu 923da6e7196b3babf5f95908d145b0bc # abaodog 384 using 512 thread, 256 using 768 thread
# ./md5_gpu 85d20affe5a0d1fb4de64fbffcfe39a3 # DeCoDeR 1556 using 764 thread