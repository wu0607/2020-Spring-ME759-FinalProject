#include <cstring>
#include <iostream>

class MD5 {
public:
  // typedef unsigned int size_type; // must be 32bit
  MD5();
  MD5(const std::string& text);
  void pipeline(const unsigned char *buf, int length);
  void pipeline(const char *buf, int length);
  MD5& finalize();
  std::string hexString() const;
  friend std::ostream& operator<<(std::ostream&, MD5 md5);

private:
  void init();
  typedef unsigned char uint1; //  8bit
  typedef unsigned int uint4;  // 32bit
  
  void processBlock(const uint1 block[64]);
  static void padding(uint4 output[], const uint1 input[], int len);
  static void encode(uint1 output[], const uint4 input[], int len);
  bool done;
  uint1 buffer[64]; // bytes that didn't fit in last 64 byte chunk
  uint4 count[2];   // 64bit counter for number of bits (lo, hi)
  uint4 state[4];   // digest so far
  uint1 digest[16]; // the result
};

std::string md5(const std::string str);