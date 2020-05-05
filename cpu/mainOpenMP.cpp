#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <math.h>
#include <vector>
#include <chrono>
#include "omp.h"

// custom library
#include "md5.h"
#include "util.h"

using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

vector<char> alphabet;

void md5_crack(string hash, string file);

int main(int argc, char* argv[]) {
	if(argc == 3 || argc == 4) {
		string type = argv[1];
		string hash = argv[2];
		string dict = "";
		if(argc == 4){
			dict = argv[3];
		}else{
			cout << "Password wordlist is not provided, generate data recurrsively" << endl;
		}
			
		std::transform(type.begin(), type.end(), type.begin(), ::toupper);

		if(type == "MD5"){
			cout << "Hashing algorithm: MD5" << endl;
			cout << "HashCode: " << hash <<endl;
			if (dict != "")	cout << "Filename: " << dict <<endl;
			
			md5_crack(hash, dict);
			
		}

	} else {
		showHelper();
	}
}

void md5_crack(string hash, string filename) {
	int tries = 0;
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
		
		cout << "Generating data from \"";
		for(int i=0; i<alphabet.size(); i++) { 
			cout << alphabet[i]; 
		}
		cout << "\" ..." << endl;
		 
		volatile bool find = false;
		long long maxVal = (long long)pow (CONST_CHARSET_LENGTH, PASSWORD_LEN);
		cout << "Total combinations: " << maxVal << endl;
		long long totalCount = 0;
		double duration_sec = 0;
		high_resolution_clock::time_point start = high_resolution_clock::now();
    	high_resolution_clock::time_point end;
		
		#pragma omp parallel for shared(find) schedule(auto) reduction(+:totalCount) num_threads(12)
		for (long long i = 0; i<maxVal; i++) {
			if (find){
				if(!duration_sec){
					end = high_resolution_clock::now(); 
					duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start).count()/1000;
				}
				continue;
			}
			totalCount += 1;
			string cand = customToString(i);
			string hash_sum = md5(cand);
			if (hash_sum == hash) {
				cout << "*** Thread" << omp_get_thread_num() << "[" << i << "] - PASSWORD FOUND - " << cand << " ***" << endl;
				// cout << "threadNum: " << omp_get_num_threads() << endl;
				cout.flush();
				find = true;
			} 
		}

		cout << "Duration = " << duration_sec << " sec" << endl;
		cout << "Throughput = " << totalCount / duration_sec << " #hash/sec" << endl;
		
	}

}