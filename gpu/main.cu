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

// custom define
#define WORD_LENGTH_MIN 1
#define WORD_LENGTH_MAX 8
#define WORD_LIMIT 10
#define CHARSET_LIMIT 64
#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_LENGTH (sizeof(CHARSET) - 1)

// hash number workload for each gpu launch
// TOTAL_BLOCKS * TOTAL_THREAD * HASHES_PER_KERNEL
// Using UL to make sure not overflow
#define TOTAL_BLOCKS 16384UL
#define TOTAL_THREAD 512UL
#define HASHES_PER_KERNEL 128UL

// #include "md5.cu"
#include "md5_unroll.cu"
using namespace std;

// Global variables 
uint8_t g_wordLen;
char g_wordArr[WORD_LIMIT];
char g_charset[CHARSET_LIMIT];
char g_crackedRes[WORD_LIMIT];

// Device variable, will use cudaMemcpyToSymbol to copy from device to host
__device__ char g_deviceCharset[CHARSET_LIMIT];
__device__ char g_deviceCracked[WORD_LIMIT];

__device__ __host__ bool next(uint8_t* length, char* word, uint32_t increment){
    int add, idx = 0;
    // increment 1
    while(increment > 0 && idx < WORD_LIMIT){
        if(idx >= *length && increment > 0){ // 1 >= 1 && 1 > 0
          increment--; // increment -> 0
        }

        add = increment + word[idx]; // add = 0
        word[idx] = add % CHARSET_LENGTH; // word[0] = 0; word[1] = 0
        increment = add / CHARSET_LENGTH; // increment = 1; increment = 0
        idx++; // 1 // 2
    }
   
    if(idx > *length){ // 2 here when encounter first boundary
        *length = idx; // update length
    }
   
    if(idx > WORD_LENGTH_MAX){
        return false;
    }
 
     return true;
}
 
__global__ void md5Crack(uint8_t wordLength, char* charsetWord, uint32_t hash01, uint32_t hash02, uint32_t hash03, uint32_t hash04){
    // calculate global thread index
    uint32_t g_idx = (blockIdx.x * blockDim.x + threadIdx.x);
    
    // Shared variables boost performance 1.16x
    __shared__ char sCharset[CHARSET_LIMIT];
    
    // Thread variables
    char threadCharsetWord[WORD_LIMIT];
    char threadTextWord[WORD_LIMIT];
    uint8_t threadWordLength;
    uint32_t tHashArray[4]; // 128-bit hash
    // uint32_t h1, h2, h3, h4; // 128-bit hash
    
    memcpy(threadCharsetWord, charsetWord, WORD_LIMIT);
    memcpy(&threadWordLength, &wordLength, sizeof(uint8_t));
    memcpy(sCharset, g_deviceCharset, sizeof(uint8_t) * CHARSET_LIMIT);
    
    // Find out which word shoud be the start word
    next(&threadWordLength, threadCharsetWord, g_idx * HASHES_PER_KERNEL);
    
    #pragma unroll
    for(uint32_t hash = 0; hash < HASHES_PER_KERNEL; hash++){
        // get current word
        for(uint32_t i = 0; i < threadWordLength; i++){
            threadTextWord[i] = sCharset[threadCharsetWord[i]];
        }

        md5Hash((unsigned char*)threadTextWord, threadWordLength, tHashArray);   
        // md5Hash((unsigned char*)threadTextWord, threadWordLength, &h1, &h2, &h3, &h4);   
        
        if(tHashArray[0] == hash01 && tHashArray[1] == hash02 && tHashArray[2] == hash03 && tHashArray[3] == hash04){
        // if(h1 == hash01 && h2 == hash02 && h3 == hash03 && h4 == hash04){
            memcpy(g_deviceCracked, threadTextWord, threadWordLength);
        }
        
        if(!next(&threadWordLength, threadCharsetWord, 1)){
            break;
        }
    }
}

void showHelper() {
    cout << "Usage: ./md5craker_gpu [HASH]" << endl;
    exit(1);
}

char hex2Bin(char ch){
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
    if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
    return 0;
}
 
void hashcode2Int(char* md5, uint *md5Hash){
    for (int i = 0; i < 32; i += 2) {
        uint A = uint(hex2Bin(md5[i]));
        uint B = uint(hex2Bin(md5[i+1]));
        uint C = A * 16 + B;
        C = C << 24;
        if(i < 8) {
            md5Hash[0] = (md5Hash[0] >> 8) | C;
        } else if (i < 16) {
            md5Hash[1] = (md5Hash[1] >> 8) | C;
        } else if (i < 24) {
            md5Hash[2] = (md5Hash[2] >> 8) | C;
        } else if(i < 32) {
            md5Hash[3] = (md5Hash[3] >> 8) | C;
        }
    }
}

int main(int argc, char* argv[]){
    if(argc != 2 || strlen(argv[1]) != 32){
        showHelper();
    }
    
    // check available gpu devices
    int devices;
    CHECK_ERROR(cudaGetDeviceCount(&devices));
    
    CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    
    cout << "Notice: " << devices << " device(s) found" << endl;
    cout << argv[1] << endl;
    
    // convert target hash to uint32 array
    uint32_t md5Hash[4];
    hashcode2Int(argv[1], md5Hash);
 
   
    // global variable init
    memset(g_wordArr, 0, WORD_LIMIT);
    memset(g_crackedRes, 0, WORD_LIMIT);
    memcpy(g_charset, CHARSET, CHARSET_LENGTH);
    g_wordLen = WORD_LENGTH_MIN;

    
    cudaSetDevice(0);
    
    // time
    cudaEvent_t startTime;
    cudaEvent_t endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime);
   
    // set different 
    char** words = new char*[devices];
    long totalCount = 0;
   
    while(true){
        bool result = false;
        bool found = false;
        
        for(int device = 0; device < devices; device++){
            CHECK_ERROR(cudaSetDevice(device));
            
            // update current data for launching kernel func
            CHECK_ERROR(cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CHARSET_LIMIT, 0, cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpyToSymbol(g_deviceCracked, g_crackedRes, sizeof(uint8_t) * WORD_LIMIT, 0, cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMalloc((void**)&words[device], sizeof(uint8_t) * WORD_LIMIT));
            CHECK_ERROR(cudaMemcpy(words[device], g_wordArr, sizeof(uint8_t) * WORD_LIMIT, cudaMemcpyHostToDevice)); // use updated g_wordArr as new start word
          
            md5Crack<<<TOTAL_BLOCKS, TOTAL_THREAD>>>(g_wordLen, words[device], md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3]);
            
            // global monitor current word
            result = next(&g_wordLen, g_wordArr, TOTAL_THREAD * HASHES_PER_KERNEL * TOTAL_BLOCKS);
            totalCount += 1;
        }

        char currentWord[WORD_LIMIT];
        for(int i = 0; i < g_wordLen; i++){
            currentWord[i] = g_charset[g_wordArr[i]];
        }
        cout << totalCount << " currently at " << string(currentWord, g_wordLen) << " word length = " << (uint32_t)g_wordLen << endl;
    
        
        for(int device = 0; device < devices; device++){
            CHECK_ERROR(cudaSetDevice(device));
            CHECK_ERROR(cudaDeviceSynchronize());
            CHECK_ERROR(cudaMemcpyFromSymbol(g_crackedRes, g_deviceCracked, sizeof(uint8_t) * WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
            // check result
            if(found = *g_crackedRes != 0){     
                cout << "===== Notice: cracked " << g_crackedRes << " =====" << endl; 
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
   
    // calculate time
    CHECK_ERROR(cudaSetDevice(0));
    cudaEventRecord(endTime);
    cudaEventSynchronize(endTime);
    float ms;
    cudaEventElapsedTime(&ms, startTime, endTime);
    cout << "Elapsed time: " << ms << " ms" << endl;
    totalCount = totalCount * TOTAL_THREAD * HASHES_PER_KERNEL * TOTAL_BLOCKS;
    cout << "totalCount: " << totalCount << " throughput= " << ((double)totalCount*1000.0/ms) << " #hash/sec" << endl;

    // free memory
    for(int device = 0; device < devices; device++){
        cudaSetDevice(device);
        cudaFree((void**)words[device]);
    }
   
    delete[] words;

    cudaEventDestroy(startTime);
    cudaEventDestroy(endTime);
}