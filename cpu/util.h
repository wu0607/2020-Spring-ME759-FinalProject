#include <iostream>
#include <vector>

using namespace std;

// all combinations of input //
extern vector<char> allDataTmp;
extern vector<string> allData;

void saveToVector(const vector<char>& v);
void getCombination(vector<char> alphabet, int k);