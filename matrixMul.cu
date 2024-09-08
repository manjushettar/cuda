#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define N 1024
__global__ void matrixMul(int *a, int *b, int *c){
	int row = threadIdx.x + blockDim.x*blockIdx.x;
	int col = threadIdx.y + blockDim.y*blockIdx.y;

	if(row < N && col < N){
		int dotProd = 0;
		for(int i = 0; i < N; i++){
			dotProd += a[row * N + i] * b[i * N + col];
		}
		c[row * N + col] = dotProd;
	}
}

void fill_matrices(int *a, int *b){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			a[i * N + j] = i;
			b[i * N + j] = i;
		}
	}
}

void check_errors(int *a, int *b, int *c){
	int *temp;
	temp = (int *)malloc(N * N * sizeof(int));	
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			for(int k = 0; k < N; k++){
				temp[i * N + j]+= a[i * N + k] * b[k * N + j];
			}
		}
	}

	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			assert(temp[i * N + j] == c[i * N + j]);
		}
	}
}

int main(){
	// 2x2 thread blocks
	// x and y dimensions -> x,y dimension of thread (row, col)
	// each thread writes to an index on the output matrix

	// coalescing writes: matrices are represented as arrays
	// a00 | a01 | a10 | a11 (2x2 matrix)
	// if we do column after column loading: we end up having staggered access patterns
	// if we do row after row loading: all of our loads are sequential (no need to calculate offsets)
	// might as well represent our own matrices as N * N arrays
	// then... we must calculate for offset row * N + col

	// in matrix mul:
	// we load all the first elements of the rows (all located in one straight line)
	// then we load the first elements of the column (diverse accesses + offsets)

	size_t size = N * N * sizeof(int);
	int *ha, *hb, *hc, *da, *db, *dc;

	dim3 threads(16,16);
	int blocks = (N + 16 - 1) / 16;
	dim3 gridSize(blocks, blocks);

	ha = (int *) malloc(size);
	hb = (int *) malloc(size);
	hc = (int *) malloc(size);

	cudaMalloc(&da, size);
	cudaMalloc(&db, size);
	cudaMalloc(&dc, size);
	
	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);
	

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	matrixMul<<<gridSize, threads>>>(da, db, dc);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Time in kernel: %f on: %d elements\n.", ms, N);
	
	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
	check_errors(ha, hb, hc);
	printf("Succesfully completed BatChest!\n");
	return 0;
}
