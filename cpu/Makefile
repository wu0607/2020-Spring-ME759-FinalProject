all: mpi openmp

mpi:
	mpicxx mainMPI.cpp md5.cpp util.cpp -o md5craker_mpi -O3
openmp:
	g++ mainOpenMP.cpp md5.cpp util.cpp -lomp -o md5craker_omp -O3
clean:
	rm md5craker *.out *.err
