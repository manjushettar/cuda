#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define N 1024
#define S_SIZE 32
#define SIZE 32

__global__ void sum_red(int *a, int *r){
	__shared__ int partial_sum[S_SIZE];
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	partial_sum[threadIdx.x] += a[i];
	__syncthreads();

	for(int s = 1; s < blockDim.x; s*=2){
		if(threadIdx.x % (2 * s) == 0){
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}
	if(threadIdx.x == 0){
		r[blockIdx.x] = partial_sum[0];
	}
}

void fill_vec(int *a){
	for(int i = 0; i < N; i++){
		a[i] = 1;
	}
}

void check_vec(int *a){
	assert(a[0] == 1024);
}
int main(){
	size_t size = N * sizeof(int);
	int *ha, *hr;
	ha = (int *)malloc(size);
	hr = (int *)malloc(size);
	fill_vec(ha);

	int *da, *dr;
	cudaMalloc(&da, size);
	cudaMalloc(&dr, size);
	
	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
	
	int threads = SIZE;
	int blocks = (int) (ceil(N / threads)); 

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	sum_red<<<blocks, threads>>>(da, dr);

	sum_red<<<1, blocks>>>(dr, dr);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Time elapsed: %f\n", ms);
	
	cudaMemcpy(hr, dr, N, cudaMemcpyDeviceToHost);	
	check_vec(hr);
	printf("BATCHEST!!\n");

	cudaFree(dr);
	cudaFree(da);
	free(ha);
	free(hr);
	return 0;
}
