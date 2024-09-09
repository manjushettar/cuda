#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#define N 1024
#define SHARED_SIZE 16 * 16 //thread blocks that are 16x16 = 256 thread pb * 4 (since we are using ints)


__global__ void matrixMulTiled(int *a, int *b, int *c, int tile_size){
	
	__shared__ int A[SHARED_SIZE];
	__shared__ int B[SHARED_SIZE]; // shared memory

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	int row = by * tile_size + ty;
	int col = bx * tile_size + tx;	

	int temp_val = 0;
	// sweep tiles over entire matrix
	for(int i = 0; i < (N / tile_size); i++){
		// every thread in a threadblock loads one elemnt into shared mem
		// The location in shared mem corresponds to thread's position in threadblock
		
		// for A: 
			// row * n indexes the global row for this thread
			// i * tile_size indexes the new set of columns each iteration
			// tx indexes the column within that set
		// for B:
			// i * tile_size *N indexes the next set of rows for iteration
			// ty * n indexes the row within the set
			// col indexes the global column
		A[(ty * tile_size) + tx] = a[row * N + (i * tile_size + tx)];
		B[(ty * tile_size) + tx] = b[(i * tile_size * N + ty * N) + col];

		__syncthreads();

		for(int j = 0; j < tile_size; j++){
			temp_val += A[ty * tile_size + j] * B[j * tile_size + tx];
		}

		__syncthreads();
	}	
	c[(row * N) + col] = temp_val;
}



void fill_matrix(int *a){
	for(int i = 0; i < N; i++){
		for(int k = 0; k < N; k++){
			a[i * N + k] = i;	
		}
	}
}

void check_answer(int *a, int *b, int *c){
	int *check;
	check = (int *)malloc(N * N * sizeof(int));
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			int temp = 0;
			for(int k = 0; k < N; k++){
				temp += a[i * N + k] * b[k * N + j];
			}
			check[i * N + j]  = temp;
		}
	}

	for(int i = 0; i < N; i++){
		for(int j =0; j < N; j++){
			assert(c[i * N + j] == check[i * N + j]);
		}
	}
}

int main(){
	// shared memory -> private per threadblock, user managed L1 cache essentialy
	// shorter cycle access time than DRAM retrieval

	// cache tiling -> when we have very large inputs, they won't fit in the cache
	// lets only cache the tile of the input that we're using 


	// every loop iteration of the input matrices, we only go to as far as the tile size

	// calculating index for loading into shared memory:
		// constant row, loop varying column
		// constant column, loop varying row

		// A[y][k] * B[k][x]
		// row is loop invariant for A (doesn't change)
		// col is loop invariant for B (doesn't change) 
	size_t size = N * N * sizeof(int);

	int *ha, *hb, *hc, *da, *db, *dc;

	ha = (int *)malloc(size);
	hb = (int *)malloc(size);
	hc = (int *)malloc(size);
	
	fill_matrix(ha);
	fill_matrix(hb);

	cudaMalloc(&da, size);
	cudaMalloc(&db, size);
	cudaMalloc(&dc, size);

	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);

	int threads = 16;
	int blocks = (N + threads - 1) / threads;
	dim3 grid_size(blocks, blocks);
	dim3 block_size(threads, threads);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	matrixMulTiled<<<grid_size, block_size>>>(da, db, dc, threads);
	cudaEventCreate(&stop);
	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(start);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Time elapsed: %f on %d elements.\n", ms, N);
	
	check_answer(ha, hb, hc);

	free(ha);
	free(hb);
	free(hc);
	
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	return 0;
}
