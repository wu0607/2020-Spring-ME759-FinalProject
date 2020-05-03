#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
// #include <device_functions.h>

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
    if(code != cudaSuccess){
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if(abort) exit(code);
    }
}

__device__ inline uint32_t F(uint32_t x, uint32_t y, uint32_t z) {
    return z ^ (x & (y ^ z));
}
 
__device__ inline uint32_t G(uint32_t x, uint32_t y, uint32_t z) {
    return y ^ (z & (x ^ y));
}
 
__device__ inline uint32_t H(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}
 
__device__ inline uint32_t I(uint32_t x, uint32_t y, uint32_t z) {
    return y ^ (x | ~z);
}
 
__device__ inline uint32_t rotate_left(uint32_t x, int n) {
    return (x << n) | (x >> (32-n));
}

__device__ inline void FF(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
    a = rotate_left(a + F(b,c,d) + x + ac, s) + b;
}

__device__ inline void GG(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
    a = rotate_left(a + G(b,c,d) + x + ac, s) + b;
}

__device__ inline void HH(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
    a = rotate_left(a + H(b,c,d) + x + ac, s) + b;
}

__device__ inline void II(uint32_t& a, uint32_t b, uint32_t c, uint32_t d, uint32_t x, uint32_t s, uint32_t ac) {
    a = rotate_left(a + I(b,c,d) + x + ac, s) + b;
}

__device__ inline void padding(uint32_t* x, unsigned char data[], uint32_t length) {
    // padding the input string
    int i = 0;
    for(i=0; i < length; i++){
        x[i / 4] |= data[i] << ((i % 4) * 8);
    }
    
    x[i / 4] |= 0x80 << ((i % 4) * 8);

    uint32_t bitlen = length * 8;
    x[14] = bitlen;
    x[15] = 0;
}

__device__ inline void md5Hash(unsigned char* data, uint32_t length, uint32_t *a1, uint32_t *b1, uint32_t *c1, uint32_t *d1){
    const uint32_t a0 = 0x67452301;
    const uint32_t b0 = 0xEFCDAB89;
    const uint32_t c0 = 0x98BADCFE;
    const uint32_t d0 = 0x10325476;

    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t d = 0;
    
    // padding the input string
    uint32_t x[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    padding(x, data, length);
    // int i = 0;
    // for(i=0; i < length; i++){ 
    //     x[i / 4] |= data[i] << ((i % 4) * 8);
    // }
    
    // x[i / 4] |= 0x80 << ((i % 4) * 8);

    // uint32_t bitlen = length * 8;
    // x[14] = bitlen;
    // x[15] = 0;

    //Initialize hash value for this chunk:
    a = a0;
    b = b0;
    c = c0;
    d = d0;

    /* Round 1 */
    #define S11 7
    #define S12 12
    #define S13 17
    #define S14 22
    FF (a, b, c, d, x[ 0], S11, 0xd76aa478); // 1
    FF (d, a, b, c, x[ 1], S12, 0xe8c7b756); // 2
    FF (c, d, a, b, x[ 2], S13, 0x242070db); // 3
    FF (b, c, d, a, x[ 3], S14, 0xc1bdceee); // 4
    FF (a, b, c, d, x[ 4], S11, 0xf57c0faf); // 5
    FF (d, a, b, c, x[ 5], S12, 0x4787c62a); // 6
    FF (c, d, a, b, x[ 6], S13, 0xa8304613); // 7
    FF (b, c, d, a, x[ 7], S14, 0xfd469501); // 8
    FF (a, b, c, d, x[ 8], S11, 0x698098d8); // 9
    FF (d, a, b, c, x[ 9], S12, 0x8b44f7af); // 10
    FF (c, d, a, b, x[10], S13, 0xffff5bb1); // 11
    FF (b, c, d, a, x[11], S14, 0x895cd7be); // 12
    FF (a, b, c, d, x[12], S11, 0x6b901122); // 13
    FF (d, a, b, c, x[13], S12, 0xfd987193); // 14
    FF (c, d, a, b, x[14], S13, 0xa679438e); // 15
    FF (b, c, d, a, x[15], S14, 0x49b40821); // 16

    /* Round 2 */
    #define S21 5
    #define S22 9
    #define S23 14
    #define S24 20
    GG (a, b, c, d, x[ 1], S21, 0xf61e2562); // 17
    GG (d, a, b, c, x[ 6], S22, 0xc040b340); // 18
    GG (c, d, a, b, x[11], S23, 0x265e5a51); // 19
    GG (b, c, d, a, x[ 0], S24, 0xe9b6c7aa); // 20
    GG (a, b, c, d, x[ 5], S21, 0xd62f105d); // 21
    GG (d, a, b, c, x[10], S22,  0x2441453); // 22
    GG (c, d, a, b, x[15], S23, 0xd8a1e681); // 23
    GG (b, c, d, a, x[ 4], S24, 0xe7d3fbc8); // 24
    GG (a, b, c, d, x[ 9], S21, 0x21e1cde6); // 25
    GG (d, a, b, c, x[14], S22, 0xc33707d6); // 26
    GG (c, d, a, b, x[ 3], S23, 0xf4d50d87); // 27
    GG (b, c, d, a, x[ 8], S24, 0x455a14ed); // 28
    GG (a, b, c, d, x[13], S21, 0xa9e3e905); // 29
    GG (d, a, b, c, x[ 2], S22, 0xfcefa3f8); // 30
    GG (c, d, a, b, x[ 7], S23, 0x676f02d9); // 31
    GG (b, c, d, a, x[12], S24, 0x8d2a4c8a); // 32

    /* Round 3 */
    #define S31 4
    #define S32 11
    #define S33 16
    #define S34 23
    HH (a, b, c, d, x[ 5], S31, 0xfffa3942); // 33
    HH (d, a, b, c, x[ 8], S32, 0x8771f681); // 34
    HH (c, d, a, b, x[11], S33, 0x6d9d6122); // 35
    HH (b, c, d, a, x[14], S34, 0xfde5380c); // 36
    HH (a, b, c, d, x[ 1], S31, 0xa4beea44); // 37
    HH (d, a, b, c, x[ 4], S32, 0x4bdecfa9); // 38
    HH (c, d, a, b, x[ 7], S33, 0xf6bb4b60); // 39
    HH (b, c, d, a, x[10], S34, 0xbebfbc70); // 40
    HH (a, b, c, d, x[13], S31, 0x289b7ec6); // 41
    HH (d, a, b, c, x[ 0], S32, 0xeaa127fa); // 42
    HH (c, d, a, b, x[ 3], S33, 0xd4ef3085); // 43
    HH (b, c, d, a, x[ 6], S34,  0x4881d05); // 44
    HH (a, b, c, d, x[ 9], S31, 0xd9d4d039); // 45
    HH (d, a, b, c, x[12], S32, 0xe6db99e5); // 46
    HH (c, d, a, b, x[15], S33, 0x1fa27cf8); // 47
    HH (b, c, d, a, x[ 2], S34, 0xc4ac5665); // 48
    
    /* Round 4 */
    #define S41 6
    #define S42 10
    #define S43 15
    #define S44 21
    II (a, b, c, d, x[ 0], S41, 0xf4292244); // 49
    II (d, a, b, c, x[ 7], S42, 0x432aff97); // 50
    II (c, d, a, b, x[14], S43, 0xab9423a7); // 51
    II (b, c, d, a, x[ 5], S44, 0xfc93a039); // 52
    II (a, b, c, d, x[12], S41, 0x655b59c3); // 53
    II (d, a, b, c, x[ 3], S42, 0x8f0ccc92); // 54
    II (c, d, a, b, x[10], S43, 0xffeff47d); // 55
    II (b, c, d, a, x[ 1], S44, 0x85845dd1); // 56
    II (a, b, c, d, x[ 8], S41, 0x6fa87e4f); // 57
    II (d, a, b, c, x[15], S42, 0xfe2ce6e0); // 58
    II (c, d, a, b, x[ 6], S43, 0xa3014314); // 59
    II (b, c, d, a, x[13], S44, 0x4e0811a1); // 60
    II (a, b, c, d, x[ 4], S41, 0xf7537e82); // 61
    II (d, a, b, c, x[11], S42, 0xbd3af235); // 62
    II (c, d, a, b, x[ 2], S43, 0x2ad7d2bb); // 63
    II (b, c, d, a, x[ 9], S44, 0xeb86d391); // 64

    a += a0;
    b += b0;
    c += c0;
    d += d0;

    *a1 = a;
    *b1 = b;
    *c1 = c;
    *d1 = d;
}