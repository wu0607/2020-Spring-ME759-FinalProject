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