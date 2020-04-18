#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include "md5.h"

using namespace std;


// Global Variables //
bool verbose;

//Function prototypes //
void md5_crack(string hash, string file);

//  / ~ ~ ~ ~ CODE ~ ~  ~ ~ \  \\
// INTERFACE //
void interface() {
	cout << "Usage: ./md5craker [HASH-TYPE] [HASH] [DICTIONARY] [-v]" << endl;
}

// MAIN //
int main(int argc, char* argv[]) {
	if(argc < 4) {
		interface();
	} else if(argc == 4 || argc == 5) {
		string type = argv[1];
		string hash = argv[2];
		string dict = argv[3];
		std::ifstream file(dict);
		if(file.is_open()) {
			if (argc == 5) {
				verbose = true;
			} else {
				verbose = false;
			}
			cout << "Dictionary: " << dict << endl;
			if(type == "MD5" || type == "md5"){
				cout << "Hashing algorithm: MD5" << endl;
				cout << "Cracking..." << endl << endl;
				md5_crack(hash, dict);
			}
			cout << "Cracking..." << endl;
		} else {
			interface();
			cout << endl << "File could not be found." << endl;
		}
	} else {
		interface();
	}
}

// CRACKING //
void md5_crack(string hash, string filename) {
	int tries = 0;
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
}