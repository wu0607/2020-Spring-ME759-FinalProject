#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_functions.h>

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

 __device__ inline void md5Hash(unsigned char* data, uint32_t length, uint32_t *a1, uint32_t *b1, uint32_t *c1, uint32_t *d1) {
     // Constants
    uint32_t md5_constants[64] = {
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

    uint32_t md5_shift_amounts[16] = {
        7, 12, 17, 22,
        5,  9, 14, 20,
        4, 11, 16, 23,
        6, 10, 15, 21
    };
    const uint32_t a0 = 0x67452301;
    const uint32_t b0 = 0xEFCDAB89;
    const uint32_t c0 = 0x98BADCFE;
    const uint32_t d0 = 0x10325476;
  
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t d = 0;
    
    uint32_t vals[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0};
 
    int i = 0;
    for(i=0; i < length; i++){
        vals[i / 4] |= data[i] << ((i % 4) * 8);
    }
   
    vals[i / 4] |= 0x80 << ((i % 4) * 8);
    uint32_t bitlen = length * 8;
    vals[14] = bitlen;
    vals[15] = 0;
 
    a = a0;
    b = b0;
    c = c0;
    d = d0;
 
    for (int i = 0; i < 64; i++) {
        int round = i >> 4;
        int bufferIdx = i;
        int shiftIdx = (round << 2) | (i & 3);
        uint32_t tmp = 0;
        switch (round)
        {
        case 0: // 0 - 15 F
            bufferIdx = i;
            tmp = F(b, c, d);
            break;
        case 1: // 16 - 31 G
            bufferIdx = (i*5 + 1) % 16;
            tmp = G(b, c, d);
            break;
        case 2: // 32 - 47 H
            bufferIdx = (i*3 + 5) % 16;
            tmp = H(b, c, d);
            break;
        case 3: // 48 - 63 I
            bufferIdx = (i*7) % 16;
            tmp = I(b, c, d);
            break;
        }
        tmp = tmp + a + md5_constants[i] + vals[bufferIdx];
        a = d;
        d = c;
        c = b;
        b = b + rotate_left(tmp, md5_shift_amounts[shiftIdx]);
    }
    a += a0;
    b += b0;
    c += c0;
    d += d0;
 
    *a1 = a;
    *b1 = b;
    *c1 = c;
    *d1 = d;
}
