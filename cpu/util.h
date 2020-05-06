#include <iostream>
#include <vector>
#define PASSWORD_LEN 6
#define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)

using namespace std;

// variable
extern vector<char> alphabet;

// function
void showHelper();
string customToString(long long val);
