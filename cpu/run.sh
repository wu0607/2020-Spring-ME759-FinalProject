#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -o run.out
#SBATCH -e run.err

make clean
make

# Run command:
# ./md5craker md5 [HASHED PASSWORD] [PATH TO THE WORDLIST FILE]
# ./md5craker md5 0acf4539a14b3aa27deeb4cbdf6e989f rockyou.txt

# michael with wordlist
./md5craker md5 0acf4539a14b3aa27deeb4cbdf6e989f rockyou.txt -v

# abcd with data generate ourself
./md5craker md5 e2fc714c4727ee9395f324cd2e7f331f  -v