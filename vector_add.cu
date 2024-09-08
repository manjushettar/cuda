#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define N 100000

__global__ void add_vec(int *a, int *b, int *c){
	int i = threadIdx.x + blockDim.x * blockIdx.x; // second term is 0 for 1d 

	if(i < N){
		c[i] = a[i] + b[i];
	}
}

void fill_vector(int *a){
	for(int i = 0; i < N; i++){
		a[i] = i;
	}
}

void error_check(int *a, int *b, int *c){
	for(int i = 0; i < N; i++){
		assert(c[i] == a[i] + b[i]);
	}
}


int main(){
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	h_a = (int*)malloc(N * sizeof(int));
	h_b = (int*)malloc(N * sizeof(int));
	h_c = (int*)malloc(N * sizeof(int));

	cudaMalloc(&d_a, N * sizeof(int));
	cudaMalloc(&d_b, N * sizeof(int));
	cudaMalloc(&d_c, N * sizeof(int));

	fill_vector(h_a);
	fill_vector(h_b);

	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

	int threads = 256;
	int block_size = (N + threads - 1)/threads;
	//start timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	add_vec<<<block_size, threads>>>(d_a, d_b, d_c);
	cudaEventRecord(stop);
	
	//end timer	
	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Time elapsed in kernel: %fms on: %d elements\n", ms, N);	
	
	error_check(h_a, h_b, h_c);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete(h_a);
	delete(h_b);
	delete(h_c);
	
	return 0;
}
