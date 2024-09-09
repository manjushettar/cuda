#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define N 100000
__global__ void vecAdd(int *a, int *b, int *c){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < N){
		c[i] = a[i] + b[i];
	}
}

void fill_vectors(int *a){
	for(int i = 0; i < N; i++){
		a[i] = i;
	}
}

void check_errors(int *a, int *b, int *c){
	for(int i = 0; i < N; i++){
		assert(c[i] == a[i] + b[i]);
	}
}

int main(){
	//prev; we malloc'd host arrays and device arrays (the latter with cudaMalloc)
	int *a, *b, *c; //let's not specify where our memory is being allocated	
	size_t size = N * sizeof(int);
	int id = cudaGetDevice(&id);	

	//let's let cuda automatically decide how to transfer the 
	cudaMallocManaged(&a, size);
	cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size); 

	fill_vectors(a);
	fill_vectors(b);

	int block_size = 256;
	int blocks = (N + block_size - 1)/block_size;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaMemPrefetchAsync(a, size, id); //prefetch the memory to the corresponding device id (async, so we can do other stuff
	cudaMemPrefetchAsync(b, size, id);

	vecAdd<<<blocks, block_size>>>(a, b, c); //the gpu has to pagefault for the cpu memory!
	cudaEventRecord(stop);

	cudaDeviceSynchronize(); //wait until all previous operations are complete
	cudaEventSynchronize(stop);

	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Kernel completed on %d elements in: %fms\n", N, ms);
	
	cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
	check_errors(a, b, c);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	printf("Completed.\n");

	return 0;
}
