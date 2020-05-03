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
// #include <device_functions.h>

// custom define
#define WORD_LIMIT 10
#define CHARSET_LIMIT 100

#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_LENGTH (sizeof(CHARSET) - 1)

#define WORD_LENGTH_MIN 1
#define WORD_LENGTH_MAX 8

// hash number workload for each gpu launch
// OTAL_BLOCKS * TOTAL_THREADS * HASHES_PER_KERNEL
#define TOTAL_BLOCKS 16384UL
#define TOTAL_THREADS 512UL
#define HASHES_PER_KERNEL 128UL

// #include "md5.cu"
#include "md5_unroll.cu"
using namespace std;

// Global variables 
uint8_t g_wordLength;
char g_word[WORD_LIMIT];
char g_charset[CHARSET_LIMIT];
char g_cracked[WORD_LIMIT];
__device__ char g_deviceCharset[CHARSET_LIMIT];
__device__ char g_deviceCracked[WORD_LIMIT];

__global__ bool next(uint8_t* length, char* word, uint32_t increment){
    int idx = 0;
    // increment 1
    while(increment > 0 && idx < WORD_LIMIT){
        if(idx >= *length && increment > 0){ // 1 >= 1 && 1 > 0
          increment--; // increment -> 0
        }

        int add = increment + word[idx]; // add = 0
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
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * HASHES_PER_KERNEL;
    
    /* Shared variables */
    __shared__ char sharedCharset[CHARSET_LIMIT];
    
    /* Thread variables */
    char threadCharsetWord[WORD_LIMIT];
    char threadTextWord[WORD_LIMIT];
    uint8_t threadWordLength;
    uint32_t threadHash01, threadHash02, threadHash03, threadHash04;
    
    /* Copy everything to local memory */
    memcpy(threadCharsetWord, charsetWord, WORD_LIMIT);
    memcpy(&threadWordLength, &wordLength, sizeof(uint8_t));
    memcpy(sharedCharset, g_deviceCharset, sizeof(uint8_t) * CHARSET_LIMIT);
    
    /* Increment current word by thread index */
    next(&threadWordLength, threadCharsetWord, idx);
    
    for(uint32_t hash = 0; hash < HASHES_PER_KERNEL; hash++){
        // get current word
        for(uint32_t i = 0; i < threadWordLength; i++){
            threadTextWord[i] = sharedCharset[threadCharsetWord[i]];
            if (idx == 0) {
                printf("[%d]%c ", threadCharsetWord[i], threadTextWord[i]);
            }
        }
        if (idx == 0) {
            printf("\n");
        }

        
        md5Hash((unsigned char*)threadTextWord, threadWordLength, &threadHash01, &threadHash02, &threadHash03, &threadHash04);   
        
        if(threadHash01 == hash01 && threadHash02 == hash02 && threadHash03 == hash03 && threadHash04 == hash04){
            memcpy(g_deviceCracked, threadTextWord, threadWordLength);
        }
        
        int tmp = threadWordLength;
        if(!next(&threadWordLength, threadCharsetWord, 1)){
            break;
        }
        if (tmp != threadWordLength && idx == 0) {
            printf("original len %d after len %d\n", tmp, threadWordLength);
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
    
    CHECK_ERROR(cudaSetDeviceFlags(cudaDeviceScheduleSpin)); // not sure use it or not
    
    cout << "Notice: " << devices << " device(s) found" << endl;
    cout << argv[1] << endl;
    
    /* Hash stored as u32 integers */
    uint32_t md5Hash[4];
    hashcode2Int(argv[1], md5Hash);
 
   
    /* Fill memory */
    memset(g_word, 0, WORD_LIMIT);
    memset(g_cracked, 0, WORD_LIMIT);
    memcpy(g_charset, CHARSET, CHARSET_LENGTH);
    
    g_wordLength = WORD_LENGTH_MIN;
    
    cudaSetDevice(0);
    
    /* Time */
    cudaEvent_t startTime;
    cudaEvent_t endTime;
    
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);
    cudaEventRecord(startTime);
   
    /* Current word is different on each device */
    char** words = new char*[devices];
    long totalCount = 0;
   
    while(true){
        bool result = false;
        bool found = false;
        
        for(int device = 0; device < devices; device++){
            CHECK_ERROR(cudaSetDevice(device));
            
            /* Copy current data */
            CHECK_ERROR(cudaMemcpyToSymbol(g_deviceCharset, g_charset, sizeof(uint8_t) * CHARSET_LIMIT, 0, cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMemcpyToSymbol(g_deviceCracked, g_cracked, sizeof(uint8_t) * WORD_LIMIT, 0, cudaMemcpyHostToDevice));
            CHECK_ERROR(cudaMalloc((void**)&words[device], sizeof(uint8_t) * WORD_LIMIT));
            CHECK_ERROR(cudaMemcpy(words[device], g_word, sizeof(uint8_t) * WORD_LIMIT, cudaMemcpyHostToDevice)); // use updated g_word as new start word
          
            /* Start kernel */
            md5Crack<<<TOTAL_BLOCKS, TOTAL_THREADS>>>(g_wordLength, words[device], md5Hash[0], md5Hash[1], md5Hash[2], md5Hash[3]);
            
            /* Global increment */
            result = next(&g_wordLength, g_word, TOTAL_THREADS * HASHES_PER_KERNEL * TOTAL_BLOCKS);
            cout << "now g_wordLength = " << (uint32_t)g_wordLength << endl;
            for(int i = 0; i < g_wordLength; i++){
                cout << g_charset[g_word[i]] << " ";
            }
            cout << endl;
            totalCount += 1;
        }

        /* Display progress */
        char word[WORD_LIMIT];
        
        for(int i = 0; i < g_wordLength; i++){
            word[i] = g_charset[g_word[i]];
        }
        
        cout << totalCount << " Notice: currently at " << string(word, g_wordLength) << " (" << (uint32_t)g_wordLength << ")" << endl;
    
        
        for(int device = 0; device < devices; device++){
            CHECK_ERROR(cudaSetDevice(device));
            
            /* Synchronize now */
            CHECK_ERROR(cudaDeviceSynchronize());
            
            /* Copy result */
            CHECK_ERROR(cudaMemcpyFromSymbol(g_cracked, g_deviceCracked, sizeof(uint8_t) * WORD_LIMIT, 0, cudaMemcpyDeviceToHost)); 
            
            /* Check result */
            if(found = *g_cracked != 0){     
                cout << "===== Notice: cracked " << g_cracked << " =====" << endl; 
                break;
            }
        }
        
        // check the result here
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
    totalCount = totalCount * TOTAL_THREADS * HASHES_PER_KERNEL * TOTAL_BLOCKS;
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