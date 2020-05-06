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

# Run command:
# ./md5craker md5 [HASHED PASSWORD] [PATH TO THE WORDLIST FILE]
# ./md5craker md5 0acf4539a14b3aa27deeb4cbdf6e989f rockyou.txt

# michael with wordlist
# ./md5craker md5 0acf4539a14b3aa27deeb4cbdf6e989f rockyou.txt 

# abcd with data generate ourself
# ./md5craker_omp md5 5501462a4c13dd55a6b236ef4550e3e4  # Erica
# mpirun -np 2 ./md5craker_mpi md5 5501462a4c13dd55a6b236ef4550e3e4 # test Erica using mpi
# ./md5craker md5 938c2cc0dcc05f2b68c4287040cfcf71 # frog
# echo -e "\n===SERIAL==="
# ./md5craker md5 e2fc714c4727ee9395f324cd2e7f331f   # test abcd using openmp
echo -e "\n===PURE OPENMP==="
./md5craker_omp md5 680380492d7edaab00c4630db4bc93e6   # test BoTtLe using openmp
echo -e "\n===PURE MPI==="
mpirun -np 2 ./md5craker_mpi md5 680380492d7edaab00c4630db4bc93e6 # test BoTtLe using mpi
# mpirun -np 2 ./md5craker_mpi md5 cb08ca4a7bb5f9683c19133a84872ca7 # ABCD
echo -e "\n===MPI + OPENMP==="
mpirun -np 2 ./md5craker_hybrid md5 680380492d7edaab00c4630db4bc93e6 # test BoTtLe using mpi
# mpirun -x LD_PRELOAD=libmpi.so -np 2 ./md5craker_hybrid md5 cb08ca4a7bb5f9683c19133a84872ca7 # test abcd using mpi
# mpirun -np 4 ./md5craker_mpi md5 fa246d0262c3925617b0c72bb20eeb1d # test 9999 using mpi

