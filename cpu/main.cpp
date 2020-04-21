#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
// #include <boost/algorithm/string.hpp>
#include <vector>
#include "md5.h"
#include "util.h"
#include "omp.h"
#define PASSWORDLEN 4

using namespace std;

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
			md5_crack(hash, dict);
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
				exit(0);
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
		
		cout << "Generating data from \"";
		for(int i=0; i<alphabet.size(); i++){ cout << alphabet[i] << " "; }
		cout << "\" ..." << endl;
		 
		#pragma omp parallel 
		{
			#pragma for schedule(dynamic)
			for (long long i = 0; i<9999999999ULL; i++) {
				// To do use max length to maximize thread util
				string cand = customToString(i);
				string hash_sum = md5(cand);
				if (hash_sum == hash) {
					cout << "[" << i << "] - PASSWORD FOUND - " << cand << endl;
					exit(0);
				} 
					
				if (i % 1000000 == 0) {
					cout << endl << omp_get_thread_num() << "completed " << i;
				}
				if (i % 100000 == 0) {
					cout << " .";
					cout.flush();
				}
			}
		}
	}

}