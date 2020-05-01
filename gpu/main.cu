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
using namespace std;

/* Global variables */
uint8_t g_wordLength;

char g_word[CONST_WORD_LIMIT];
char g_charset[CONST_CHARSET_LIMIT];
char g_cracked[CONST_WORD_LIMIT];

__device__ char g_deviceCharset[CONST_CHARSET_LIMIT];
__device__ char g_deviceCracked[CONST_WORD_LIMIT];

__device__ __host__ bool next(uint8_t* length, char* word, uint32_t increment){
   int idx = 0;
  
   while(increment > 0 && idx < CONST_WORD_LIMIT){
       if(idx >= *length && increment > 0){
         increment--;
       }

       int add = increment + word[idx];
       word[idx] = add % CONST_CHARSET_LENGTH;
       increment = add / CONST_CHARSET_LENGTH;
       idx++;
   }
  
   if(idx > *length){
       *length = idx;
   }
  
   if(idx > CONST_WORD_LENGTH_MAX){
       return false;
   }

    return true;
}

__global__ void md5Crack(uint8_t wordLength, char* charsetWord, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04){
   uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASHES_PER_KERNEL;
   
   /* Shared variables */
   __shared__ char sharedCharset[CONST_CHARSET_LIMIT];
   
   /* Thread variables */
   char threadCharsetWord[CONST_WORD_LIMIT];
   char threadTextWord[CONST_WORD_LIMIT];
   uint8_t threadWordLength;
   uint32_t threadHash01, threadHash02, threadHash03, threadHash04;
   
   /* Copy everything to local memory */
   memcpy(threadCharsetWord, charsetWord, CONST_WORD_LIMIT);
   memcpy(&threadWordLength, &wordLength, sizeof(uint8_t));
   memcpy(sharedCharset, g_deviceCharset, sizeof(uint8_t) * CONST_CHARSET_LIMIT);
   
   /* Increment current word by thread index */
   next(&threadWordLength, threadCharsetWord, idx);
   
   for(uint32_t hash = 0; hash < HASHES_PER_KERNEL; hash++){
       for(uint32_t i = 0; i < threadWordLength; i++){
           threadTextWord[i] = sharedCharset[threadCharsetWord[i]];
       }
       
       md5Hash((unsigned char*)threadTextWord, threadWordLength, &threadHash01, &threadHash02, &threadHash03, &threadHash04);   
   
       if(threadHash01 == hash01 && threadHash02 == hash02 && threadHash03 == hash03 && threadHash04 == hash04){
           memcpy(g_deviceCracked, threadTextWord, threadWordLength);
       }
       
       if(!next(&threadWordLength, threadCharsetWord, 1)){
           break;
       }
   }
}

int main(int argc, char* argv[]){
   /* Check arguments */
   if(argc != 2 || strlen(argv[1]) != 32){
       cout << argv[0] << " <md5_hash>" << endl;
       return -1;
   }
   
   /* Amount of available devices */
   int devices;
   CHECK_ERROR(cudaGetDeviceCount(&devices));
   
   /* Sync type */
   CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
   
   /* Display amount of devices */
   cout << "Notice: " << devices << " device(s) found" << endl;
   cout << argv[1] << endl;
   
   /* Hash stored as u32 integers */
   uint32_t md5Hash[4];
   
   /* Fill memory */
   memset(g_word, 0, CONST_WORD_LIMIT);
   memset(g_cracked, 0, CONST_WORD_LIMIT);
   memcpy(g_charset, CONST_CHARSET, CONST_CHARSET_LENGTH);
   
   /* Current word length = minimum word length */
   g_wordLength = CONST_WORD_LENGTH_MIN;
   
   /* Main device */
   cudaSetDevice(0);
   
   /* Time */
   cudaEvent_t clockBegin;
   cudaEvent_t clockLast;
   
   cudaEventCreate(&clockBegin);
   cudaEventCreate(&clockLast);
   cudaEventRecord(clockBegin, 0);
  
   /* Current word is different on each device */
   char** words = new char*[devices];
   
   for(int device = 0; device < devices; device++){
       cudaSetDevice(device);
       
       /* Copy to each device */
       CHECK_ERROR(cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CONST_CHARSET_LIMIT, 0, cudaMemcpyHostToDevice));
       CHECK_ERROR(cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyHostToDevice));
       
       /* Allocate on each device */
       CHECK_ERROR(cudaMalloc((void**)&words[device], sizeof(uint8_t) * CONST_WORD_LIMIT));
   }
  
   while(true){
       bool result = false;
       bool found = false;
       
       for(int device = 0; device < devices; device++){
           cudaSetDevice(device);
           
           /* Copy current data */
           CHECK_ERROR(cudaMemcpy(words[device], g_word, sizeof(uint8_t) * CONST_WORD_LIMIT, cudaMemcpyHostToDevice)); 
         
           /* Start kernel */
           md5Crack<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(g_wordLength, words[device], md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3]);
           
           /* Global increment */
           result = next(&g_wordLength, g_word, TOTAL_THREADS * HASHES_PER_KERNEL * TOTAL_BLOCKS);
           cudaSetDevice(device);
           
           /* Synchronize now */
           cudaDeviceSynchronize();
           
           /* Copy result */
           CHECK_ERROR(cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * CONST_WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
           
           /* Check result */
           if(found = *g_cracked != 0){     
               cout << "Notice: cracked " << g_cracked << endl; 
               break;
           }
       }
     
       if(!result || found){
           if(!result && !found){
             cout << "Notice: found nothing (host)" << endl;
           }
           
           break;
       }
   }
  
   for(int device = 0; device < devices; device++){
       cudaSetDevice(device);
       
       /* Free on each device */
       cudaFree((void**)words[device]);
   }
  
   /* Free array */
   delete[] words;
   
   /* Main device */
   cudaSetDevice(0);
   
   float milliseconds = 0;
   
   cudaEventRecord(clockLast, 0);
   cudaEventSynchronize(clockLast);
   cudaEventElapsedTime(&milliseconds, clockBegin, clockLast);
   
   cout << "Computation time " << milliseconds << " ms" << endl;
   
   cudaEventDestroy(clockBegin);
   cudaEventDestroy(clockLast);
}