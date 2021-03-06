#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH --nodes=2 --cpus-per-task=20 --ntasks-per-node=1
#SBATCH -J run
#SBATCH -o run.out
#SBATCH -e run.err

module load mpi/openmpi
module load cuda
# make clean
make

HASH_TO_CRACK=e2fc714c4727ee9395f324cd2e7f331f # abcd

echo -e "\n===SERIAL==="
./md5craker md5 $HASH_TO_CRACK   # test abcd using openmp
echo -e "\n===PURE OPENMP==="
./md5craker_omp md5 $HASH_TO_CRACK   # test abcd using openmp
echo -e "\n===PURE MPI==="
mpirun -np 2 ./md5craker_mpi md5 $HASH_TO_CRACK # abcd using mpi
echo -e "\n===MPI + OPENMP==="
mpirun -np 2 ./md5craker_hybrid md5 $HASH_TO_CRACK # test abcd using mpi + openmp

# 5501462a4c13dd55a6b236ef4550e3e4 Erica
# e2fc714c4727ee9395f324cd2e7f331f abcd 
# 938c2cc0dcc05f2b68c4287040cfcf71 frog
# fa246d0262c3925617b0c72bb20eeb1d 9999