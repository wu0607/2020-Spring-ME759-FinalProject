#include "md5.h"
#include <cstdio>
 
// this code is mainly reference from https://tools.ietf.org/html/rfc1321

// Constants
uint32_t md5Constants[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,
    0xa8304613,0xfd469501,0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,
    0x6b901122,0xfd987193,0xa679438e,0x49b40821,0xf61e2562,0xc040b340,
    0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,
    0x676f02d9,0x8d2a4c8a,0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,
    0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,0x289b7ec6,0xeaa127fa,
    0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,
    0xffeff47d,0x85845dd1,0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,
    0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

uint32_t md5ShiftAmounts[16] = {
    7, 12, 17, 22,
    5,  9, 14, 20,
    4, 11, 16, 23,
    6, 10, 15, 21
};
 
// MD5 Low Level Operations
inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) {
  return z ^ (x & (y ^ z));
}
 
inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) {
  return y ^ (z & (x ^ y));
}
 
inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) {
  return x ^ y ^ z;
}
 
inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) {
  return y ^ (x | ~z);
}
 
inline uint32_t rotate_left(uint32_t x, int n) {
  return (x << n) | (x >> (32-n));
}
 
inline uint32_t FF(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t shift, uint32_t ac) {
  return b + rotate_left(a+ F(b,c,d) + x + ac, shift);
}
 
inline uint32_t GG(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t shift, uint32_t ac) {
  return b + rotate_left(a + G(b,c,d) + x + ac, shift);
}
 
inline uint32_t HH(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t shift, uint32_t ac) {
  return b + rotate_left(a + H(b,c,d) + x + ac, shift);
}
 
inline uint32_t II(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t shift, uint32_t ac) {
  return b + rotate_left(a + I(b,c,d) + x + ac, shift);
}

/**
 * constructor to calculate md5 hash directly
 */
MD5::MD5(const std::string &text)
{
  ////////////////
  // init
  ////////////////
  done = false;
 
  memset(buffer, 0, sizeof(buffer));
  memset(count, 0, sizeof(count));
  
  // md5 magic initial number
  state[0] = 0x67452301;
  state[1] = 0xefcdab89;
  state[2] = 0x98badcfe;
  state[3] = 0x10325476;

  ////////////////
  // pipeline
  ////////////////
  pipeline(text.c_str(), text.length());


  ////////////////
  // finish
  ////////////////
  static unsigned char padding[64] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };
 
  if (!done) {
    int padLen;
    unsigned char bits[8];
    encode(bits, count, 8);
 
    // pad to 56 mod 64
    int idx = count[0] / 8 % 64;

    if (idx < 56) {
      padLen = 56 - idx;
    } else {
      padLen = 120 - idx;
    }
    pipeline(padding, padLen);
    pipeline(bits, 8);
 
    // digest should be little endian
    encode(digest, state, 16);
 
    done = true;
  }
}
 
void MD5::padding(unsigned int output[], const unsigned char input[], int len)
{
  for (int i = 0, j = 0; j < len; i++, j += 4)
    output[i] = ((unsigned int)input[j]) | (((unsigned int)input[j+1]) << 8) |
      (((unsigned int)input[j+2]) << 16) | (((unsigned int)input[j+3]) << 24);
}
 
// encode input into little endian
void MD5::encode(unsigned char output[], const unsigned int input[], int len)
{
  for (int i = 0, j = 0; j < len; i++, j += 4) {
    output[j]   = input[i] & 0xff;
    output[j+1] = (input[i] >> 8) & 0xff;
    output[j+2] = (input[i] >> 16) & 0xff;
    output[j+3] = (input[i] >> 24) & 0xff;
  }
}
 
void MD5::processBlock(const unsigned char block[64])
{
  uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];

  // first pad the block to a multiple of 512 bits
  padding(x, block, 64); // extract block into x

  // main loop
  for (int i = 0; i < 64; i++) {
    int round = i >> 4;
    int bufferIdx = i;
    int shiftIdx = (round << 2) | (i & 3);
    uint32_t tmp = 0;
    switch (round) {
      case 0: // 0 - 15 FF
        tmp = FF(a, b, c, d, x[bufferIdx], md5ShiftAmounts[shiftIdx], md5Constants[i]);
        break;
      case 1: // 16 - 31 GG
        bufferIdx = (i*5 + 1) % 16;
        tmp = GG(a, b, c, d, x[bufferIdx], md5ShiftAmounts[shiftIdx], md5Constants[i]);
        break;
      case 2: // 32 - 47 HH
        bufferIdx = (i*3 + 5) % 16;
        tmp = HH(a, b, c, d, x[bufferIdx], md5ShiftAmounts[shiftIdx], md5Constants[i]);
        break;
      case 3: // 48 - 63 II
        bufferIdx = (i*7) % 16;
        tmp = II(a, b, c, d, x[bufferIdx], md5ShiftAmounts[shiftIdx], md5Constants[i]);
        break;
    }
    a = d;
    d = c;
    c = b;
    b = tmp;
  }

  // add to state
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
} 
 
// MD5 block pipeline operation, can deal with arbitray length of msg
void MD5::pipeline(const unsigned char input[], int length)
{
  int i = 0;
  int idx = (count[0] >> 3);
  idx = (idx >= 64 ? idx % 64 : idx);
 
  // get number of bits
  count[0] += (length << 3);
  if (count[0] < (length << 3))
    count[1]++;
  count[1] += (length >> 29);
 
  // get number of bytes for filling buffer
  int front = 64 - idx;
 
  // process all the blocks, chunksize: 64
  if (length >= front){
    memcpy(&buffer[idx], input, front);
    processBlock(buffer);
 
    for (i = front; i + 64 <= length; i += 64)
      processBlock(&input[i]);
 
    idx = 0;
  }
 
  // remain input
  memcpy(&buffer[idx], &input[i], length-i);
}
 
// function overloading for both signed & unsigned char input
void MD5::pipeline(const char input[], int length)
{
  pipeline((const unsigned char*)input, length);
}
 
/**
 * return hex of digest with string
 */
std::string MD5::hex2String() const
{
  if (!done)
    return "";
 
  // initialize a c-string style buffer and initialize to 0
  char buf[33];
  memset(buf, '0', sizeof(buf));

  for (int i=0; i<16; i++) {
    // parse with hex here
    sprintf(buf + i*2, "%02x", digest[i]);
  }
 
  return std::string(buf);
}
 
/**
 * a helper function to convenienty get the md5 hash result
 *
 * will instantiate a instance of MD5 class, and return the hash
 */
std::string md5(const std::string str)
{
  MD5 md5 = MD5(str);
  return md5.hex2String();
}
