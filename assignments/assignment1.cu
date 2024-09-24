#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>
__global__ void addVector(float *a, float *b, float *c, size_t s){
    int row = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < s){
        a[row] = b[row] + c[row];
    }
}

void init_vector(float *a, size_t s){
    for(size_t i = 0; i < s; ++i){
        a[i] = 1.0 * i;
    }
}

bool checkAdd(float *a, float *b, float *c, size_t s){
    for(size_t i = 0; i < s; ++i){
        if(a[i] != b[i] + c[i]) return false;
    }

    std::cout << "Kernel size " << std::to_string(s) << " has been checked." << std::endl;
    return true;
}

int main(){
    for(size_t size = 1024; size < 20480; size*=2){
        float avgMs;
        for(int temp = 0; temp < 5; temp++){
            float *a, *b, *c;
            float *da, *db, *dc;

            size_t mallocSizes = size * sizeof(float);

            a = (float *)malloc(mallocSizes);
            b = (float *)malloc(mallocSizes);
            c = (float *)malloc(mallocSizes);

            cudaMalloc(&da, mallocSizes);
            cudaMalloc(&db, mallocSizes);
            cudaMalloc(&dc, mallocSizes);

            init_vector(b, size);
            init_vector(c, size);

            cudaMemcpy(db, b, mallocSizes, cudaMemcpyHostToDevice);
            cudaMemcpy(dc, c, mallocSizes, cudaMemcpyHostToDevice);
            
            dim3 tpb(1);
            dim3 bpg(size);
            
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            addVector<<<bpg, tpb>>>(da, db, dc, size);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);

            float ms = 0.f;
            cudaEventElapsedTime(&ms, start, stop);

            cudaMemcpy(a, da, mallocSizes, cudaMemcpyDeviceToHost);
            checkAdd(a, b, c, size);
            avgMs += ms;
        }
        avgMs /= 5;
        printf("Average Kernel time for size: %zu: %.3f\n", size, avgMs);
    }
    return EXIT_SUCCESS;
}

