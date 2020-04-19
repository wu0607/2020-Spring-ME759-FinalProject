#include "util.h"
#include <string>

void saveToVector(const vector<char>& v) {
  	static int count = 0;
  	string str = "";
  	for (int i = 0; i < v.size(); ++i) { 
	  	str += v[i];
	}
	allData.push_back(str);
}

// dfs
void getCombination(vector<char> alphabet, int k) {
	if (k == 0) {
	  	saveToVector(allDataTmp);
	  	return;
	}
	for (int i = 0; i < alphabet.size(); ++i) {
	 	allDataTmp.push_back(alphabet[i]);
	 	getCombination(alphabet, k-1);
	 	allDataTmp.pop_back();
	}
}


string customToString(long long val) { 
    string ret;
    static const size_t size = alphabet.size() -1;
    while (val) { 
        ret += alphabet[val % size];
        val /= size;
    }
    return ret;
}