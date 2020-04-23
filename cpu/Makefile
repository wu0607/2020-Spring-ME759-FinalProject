all: md5

md5:
	mpicxx main.cpp md5.cpp util.cpp -o md5craker -O3
	# g++ main.cpp md5.cpp util.cpp -fopenmp -o md5craker -O3
clean:
	rm md5craker *.out *.err
