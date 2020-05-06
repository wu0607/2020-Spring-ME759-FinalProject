# MD5 Hash Attack
## Problem statement
Exhaustive search attack on the MD5-crypt password hashing scheme using modern CPU and GPU in parallel. Measure throughput of different cracking methods in #hash/second.

[Project Proposal](https://docs.google.com/document/d/1eyJ1p6AepYZB12CC3kC60QPogzrOMCvzncfTbFjLrtE/edit?usp=sharing)

## Quick Start on euler
CPU version (Serial, OpenMP, MPI, Hybrid)
```
cd cpu
vim run.sh # modify md5hash you want to crack
sbatch run.sh
```

GPU version
```
cd gpu
vim run.sh # modify md5hash you want to crack
sbatch run.sh
```