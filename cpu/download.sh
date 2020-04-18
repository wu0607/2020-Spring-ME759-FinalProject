#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -o download.out
#SBATCH -e download.err

wget https://github.com/praetorian-code/Hob0Rules/raw/master/wordlists/rockyou.txt.gz
gzip -d rockyou.txt.gz