#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include "md5.h"
#include "util.h"
#define PASSWORDLEN 4

using namespace std;

// all combinations of input //
vector<char> allDataTmp;
vector<string> allData;

// Global Variables //
bool verbose;

//Function prototypes //
void md5_crack(string hash, string file);

//  / ~ ~ ~ ~ CODE ~ ~  ~ ~ \  \\
// showHelper //
void showHelper() {
	cout << "Usage: ./md5craker [HASH-TYPE] [HASH] [DICTIONARY] [-v]" << endl;
}

// MAIN //
int main(int argc, char* argv[]) {
	if(argc < 4) {
		showHelper();
	} else if(argc == 4 || argc == 5) {
		string type = argv[1];
		string hash = argv[2];
		string dict = argv[3];
		ifstream file(dict);
		if(file.is_open()) {
			if (argc == 5) {
				verbose = true;
			} else {
				verbose = false;
			}
		}else{
			cout << "File not exist, generate data recurrsively" << endl;
			dict = "";
		}
			
		boost::to_upper(type);
		if(type == "MD5"){
			cout << "Hashing algorithm: MD5" << endl;
			cout << "HashCode: " << hash <<endl;
			cout << "Filename: " << dict<<endl;
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
				if (verbose == true) {
					cout << "[" << tries << "] - FAILED ATTEMPT - " << pass << endl;
				}
			}
		}
	}else{
		vector<char> alphabet;
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
		getCombination(alphabet, PASSWORDLEN);
		cout << "Password with length " << PASSWORDLEN << " has " << allData.size() << " of combinations" << endl;
		
		cout << "Cracking ..." << endl << endl;
		while(tries < allData.size()) {
			if(allData[tries][0] == 'm'){
				cout << allData[tries] << endl;
			}
			string hash_sum = md5(allData[tries]);
			if (hash_sum == hash) {
				cout << "[" << tries << "] - PASSWORD FOUND - " << allData[tries] << endl;
				exit(0);
			} else {
				if (verbose == true) {
					cout << "[" << tries << "] - FAILED ATTEMPT - " << allData[tries] << endl;
				}
			}
			tries++;
		}
		
	}
}