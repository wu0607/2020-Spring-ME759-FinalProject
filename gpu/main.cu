#include <stdio.h>
 #include <iostream>
 #include <time.h>
 #include <string.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <sstream>

 #include <cuda_runtime.h>
 #include <cuda_runtime_api.h>
 #include <curand_kernel.h>
 #include <device_functions.h>

 #define CONST_WORD_LIMIT 10
 #define CONST_CHARSET_LIMIT 100

 #define CONST_CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
 #define CONST_CHARSET_LENGTH (sizeof(CONST_CHARSET) - 1)

 #define CONST_WORD_LENGTH_MIN 1
 #define CONST_WORD_LENGTH_MAX 8

 #define TOTAL_BLOCKS 16384UL
 #define TOTAL_THREADS 512UL
 #define HASHES_PER_KERNEL 128UL

 #include "md5.cu"

 #define ERROR_CHECK(X) { gpuAssert((X), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true){
  if(code != cudaSuccess){
    std::cout << "Error: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if(abort){
      exit(code);
    }
  }
}

 int main(int argc, char* argv[]){
   /* Check arguments */
   if(argc != 2 || strlen(argv[1]) != 32){
     std::cout << argv[0] << " <md5_hash>" << std::endl;
     return -1;
   }

   int devices;
   ERROR_CHECK(cudaGetDeviceCount(&devices));

   ERROR_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

   std::cout << devices << " device(s) found" << std::endl;
   std::cout << argv[1] << std::endl;

   memset(g_word, 0, CONST_WORD_LIMIT);
   memset(g_cracked, 0, CONST_WORD_LIMIT);
   memcpy(g_charset, CONST_CHARSET, CONST_CHARSET_LENGTH);

   g_wordLength = CONST_WORD_LENGTH_MIN;

   cudaSetDevice(0);

 }