#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -o run.out
#SBATCH -e run.err

make clean
make

# Run command:
# ./md5craker md5 [HASHED PASSWORD] [PATH TO THE WORDLIST FILE]
# ./md5craker md5 0acf4539a14b3aa27deeb4cbdf6e989f rockyou.txt

./md5craker md5 0acf4539a14b3aa27deeb4cbdf6e989f rockyou.txt -v