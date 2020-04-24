#include "util.h"
#include <string>

void showHelper() {
	cout << "Usage: ./md5craker [HASH-TYPE] [HASH] [DICTIONARY] [-v]" << endl;
}

string customToString(long long val) { 
    string ret;
    static const size_t size = alphabet.size();
    while (val) { 
        ret += alphabet[val % size];
        val /= size;
    }
    return ret;
}