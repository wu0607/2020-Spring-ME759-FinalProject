#include <cstring>
#include <iostream>

class MD5 {
public:
  MD5(const std::string& text);
  void pipeline(const unsigned char *buf, int length);
  void pipeline(const char *buf, int length);
  std::string hex2String() const;

private:
  void processBlock(const unsigned char block[64]);
  static void padding(unsigned int output[], const unsigned char input[], int len);
  static void encode(unsigned char output[], const unsigned int input[], int len);
  
  bool done;
  unsigned int count[2];   // 64bit counter for number of bits (lo, hi)
  unsigned int state[4];   // digest so far
  unsigned char digest[16]; // the result
  unsigned char buffer[64]; // bytes that didn't fit in last 64 byte chunk
};

// helper function
std::string md5(const std::string str);
