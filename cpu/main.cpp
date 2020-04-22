#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
// #include <boost/algorithm/string.hpp>
#include <vector>
#include <chrono>
#include "md5.h"
#include "util.h"
#include "omp.h"
#define PASSWORD_LEN 5
#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)


using namespace std;
using chrono::high_resolution_clock;
using chrono::duration;

// all combinations of input //
vector<char> allDataTmp;
vector<string> allData;
vector<char> alphabet;

//Function prototypes //
void md5_crack(string hash, string file);

// showHelper //
void showHelper() {
	cout << "Usage: ./md5craker [HASH-TYPE] [HASH] [DICTIONARY] [-v]" << endl;
}

int main(int argc, char* argv[]) {
	// timing variable
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

	if(argc == 3 || argc == 4) {
		string type = argv[1];
		string hash = argv[2];
		string dict = "";
		if(argc == 4){
			dict = argv[3];
		}else{
			cout << "File not exist, generate data recurrsively" << endl;
		}
			
		// boost::to_upper(type);
		std::transform(type.begin(), type.end(), type.begin(), ::toupper);

		if(type == "MD5"){
			cout << "Hashing algorithm: MD5" << endl;
			cout << "HashCode: " << hash <<endl;
			cout << "Filename: " << dict<<endl;
			// Todo time measurement
			start = high_resolution_clock::now(); // Get the starting timestamp
			md5_crack(hash, dict);
			end = high_resolution_clock::now(); // Get the ending timestamp
			duration_sec = std::chrono::duration_cast<duration<double, std::milli> >(end - start);
			cout << duration_sec.count()/1000 << " sec" << endl; // ms
		}

	} else {
		showHelper();
	}
}

// CRACKING //
void md5_crack(string hash, string filename) {
	int tries = 0;
	if(filename != ""){
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
	}else{
		for (char c = 'A'; c <= 'Z'; c++) { 
  			alphabet.push_back(c);
    	}  
		for (char c = 'a'; c <= 'z'; c++) { 
  			alphabet.push_back(c);
    	}  
		for (int i=0; i<=9; i++) { 
  			alphabet.push_back('0' + i);
    	}  
		
		cout << "vector size:" << alphabet.size() << CONST_CHARSET_LENGTH << endl;
		cout << "Generating data from \"";
		for(int i=0; i<alphabet.size(); i++){ cout << alphabet[i] << " "; }
		cout << "\" ..." << endl;
		 
		volatile bool find = false;
		long long maxVal = (long long)pow (CONST_CHARSET_LENGTH, PASSWORD_LEN);
		#pragma omp parallel for shared(find) schedule(dynamic)
		for (long long i = 0; i<maxVal; i++) {
			if (find){
				continue;
			}
			// To do use max length to maximize thread util
			string cand = customToString(i);
			string hash_sum = md5(cand);
			if (hash_sum == hash) {
				cout << "T" << omp_get_thread_num() << "[" << i << "] - PASSWORD FOUND - " << cand << endl;
				cout.flush();
				find = true;
			} 
				
			// if (i % 10000000 == 0) {
			// 	cout << endl << omp_get_thread_num() << "completed " << i;
			// }
			// if (i % 5000000 == 0) {
			// 	cout << " .";
			// 	cout.flush();
			// }
		}
		
	}

}