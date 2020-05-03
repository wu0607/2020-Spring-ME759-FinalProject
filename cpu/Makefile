all: pure mpi openmp hybrid

pure:
	clang++ -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp main.cpp md5_unroll.cpp util.cpp -O3 -o md5craker
mpi:
	mpicxx -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib mainMPI.cpp md5_unroll.cpp util.cpp -lomp -o md5craker_mpi -O3
openmp:
	# g++ mainOpenMP.cpp md5.cpp util.cpp -fopenmp -o md5craker_omp -O3
	clang++ -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib -lomp mainOpenMP.cpp md5_unroll.cpp util.cpp -o md5craker_omp -O3
hybrid:
	mpicxx -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib mainHybrid.cpp md5_unroll.cpp util.cpp -lomp -o md5craker_hybrid -O3
clean:
	rm md5craker_omp md5craker_mpi *.out *.err
