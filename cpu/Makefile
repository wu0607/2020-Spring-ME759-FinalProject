all: mpi openmp

mpi:
	mpicxx -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib mainMPI.cpp md5.cpp util.cpp -lomp -o md5craker_mpi -O3
openmp:
	# g++ mainOpenMP.cpp md5.cpp util.cpp -fopenmp -o md5craker_omp -O3
	clang++ -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp mainOpenMP.cpp md5.cpp util.cpp -o md5craker_omp -O3
clean:
	rm md5craker_omp md5craker_mpi *.out *.err
