#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <math.h>
#include <vector>
#include <chrono>
#include <mpi.h>
#include "omp.h"


// custom library
#include "md5.h"
#include "util.h"

using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

vector<char> alphabet;
double startTime;
long long totalCount = 0;
int tag = 1;

void md5_crack(string hash, string file);

int main(int argc, char* argv[]) {
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(argc == 3 || argc == 4) {
		string type = argv[1];
		string hash = argv[2];
		string dict = "";
		if(argc == 4){
			dict = argv[3];
		}else if(rank == 0){
			cout << "Password wordlist is not provided, generate data recurrsively" << endl;
		}
			
		std::transform(type.begin(), type.end(), type.begin(), ::toupper);

		if(type == "MD5"){
			if(rank == 0){
				cout << "Hashing algorithm: MD5" << endl;
				cout << "HashCode: " << hash <<endl;
				if (dict != "")	cout << "Filename: " << dict <<endl;
			}
			
			md5_crack(hash, dict);
			
		}

	} else {
		showHelper();
	}
}


void run_MPI(int rank, int size, int maxVal, string hash){
	// long long localCount = 0;
	// long long start = maxVal / size * rank;
	// long long end = maxVal / size * (rank + 1);
	
	// startTime = MPI_Wtime();
	// for (long long i = start; i < end; i++) {
	// 	string cand = customToString(i);
	// 	string hash_sum = md5(cand);
	// 	localCount += 1;
	// 	MPI_Allreduce(&localCount, &totalCount, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
		
	// 	if (hash_sum == hash) {
	// 		cout << "Rank" << rank << "[" << i << "] - PASSWORD FOUND - " << cand << endl;
	// 		double duration_sec = MPI_Wtime() - startTime;
	// 		cout << duration_sec << "sec" << endl;
	// 		cout.flush();
	// 		int err;
	// 		cout << "localCount:" << localCount << " ;threadNum: " << omp_get_num_threads() << endl;
	// 		cout << "totalCount:" << totalCount << " throughput= " << totalCount / duration_sec << " #hash/sec" << endl;
	// 		MPI_Abort(MPI_COMM_WORLD, err);
	// 	} 
	// }

	long long localCount = 0;
	long long start = maxVal / size * rank;
	long long end = maxVal / size * (rank + 1);
	int find = 0;
	volatile double duration_sec = 0;
	int flag = 0; 
	MPI_Request request;
	
    startTime = MPI_Wtime();
	for (long long i = start; i < end; i++) {
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG , MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);

		if(find){
			break;
		}
		if(flag){
			break;
		}
		string cand = customToString(i);
		string hash_sum = md5(cand);
		localCount += 1;
		if (hash_sum == hash) {
			cout << "*** Rank" << rank << "[" << i << "] - PASSWORD FOUND - " << cand << " ***" << endl;
			duration_sec = MPI_Wtime() - startTime;
			cout.flush();
			cout << "localCount:" << localCount << " ;threadNum: " << omp_get_num_threads() << endl;
			cout << "duration_sec:" << duration_sec << endl;
			find = 1;
			for (int r = 0; r < size; ++r) {
				if(r != rank){
           			MPI_Isend(&find, 1, MPI_INT, r, tag , MPI_COMM_WORLD, &request);
				}
        	}
		} 
	}
	
	MPI_Allreduce(&localCount, &totalCount, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	// cout << rank << " - localCount:" << localCount << " totalCount: " << totalCount << " duration_sec:" << duration_sec << " throughput= " << totalCount / duration_sec << " #hash/sec" << endl;
	if(duration_sec){
		cout << "Duration = " << duration_sec << " sec" << endl;
		cout << "Throughput = " << totalCount / duration_sec << " #hash/sec" << endl;
	}
}

void md5_crack(string hash, string filename) {
	int tries = 0;
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(filename != ""){ // direct dictionary mapping
		cout << "Cracking..." << endl << endl;
		std::ifstream file(filename);
		string pass;
		while(file >> pass) {
			tries++;
			string hash_sum = md5(pass);
			if (hash_sum == hash) {
				cout << "[" << tries << "] - PASSWORD FOUND - " << pass << endl;
				return;
			} else {
				if (tries % 1000000 == 0) {
					cout << "[" << tries << "] - FAILED ATTEMPT - " << pass << endl;
				}
			}
		}
	}else{ // brute force
		for (char c = 'A'; c <= 'Z'; c++) { 
  			alphabet.push_back(c);
    	}  
		for (char c = 'a'; c <= 'z'; c++) { 
  			alphabet.push_back(c);
    	}  
		for (int i=0; i<=9; i++) { 
  			alphabet.push_back('0' + i);
    	}  
		
		long long maxVal = (long long)pow (CONST_CHARSET_LENGTH, PASSWORD_LEN);
		if(rank == 0) {
			cout << "MPI rank size: " << size << endl;
			cout << "Generating data from \"";
			for(int i=0; i<alphabet.size(); i++) { 
				cout << alphabet[i]; 
			}
			cout << "\" ..." << endl;
			cout << "Total combinations: " << maxVal << endl;
		}

		run_MPI(rank, size, maxVal, hash);
		MPI_Finalize();
	}

}
