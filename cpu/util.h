#include <iostream>
#include <vector>

using namespace std;

// all combinations of input //
extern vector<char> allDataTmp;
extern vector<string> allData;
extern vector<char> alphabet;

void saveToVector(const vector<char>& v);
void getCombination(vector<char> alphabet, int k);
string customToString(long long val);
