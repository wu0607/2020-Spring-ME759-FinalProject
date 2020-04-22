#include <iostream>
#include <vector>
#define PASSWORD_LEN 5
#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)

using namespace std;

// all combinations of input //
extern vector<char> allDataTmp;
extern vector<string> allData;
extern vector<char> alphabet;

void saveToVector(const vector<char>& v);
void getCombination(vector<char> alphabet, int k);
string customToString(long long val);
