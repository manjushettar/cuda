#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define SHARED_SIZE 16 * 16
__global__ void gemm_v0(size_t m, size_t n, size_t k,
			const float *A, const float *B, float *C, const float alpha){
	
	// A = m x k
	// B = k x n	
	// C = m x n

	// k = stride for accessing elements in row for A since A is m x k
	// n = stride for accessing elements in col for B since B is k x n
	// n = stride for accessing elements for C since C is m x n

	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < m && col < n){
		float dotProd = 0.0;
		for(int i = 0; i < k; i++){
			dotProd += A[row * k + i] * B[i * n + col];
		}
		C[row * n + col] = alpha * dotProd;
	}
}

__global__ void gemm_v1_global(size_t m, size_t n, size_t k,
				const float *A, const float *B, float *C, const float alpha){
	// wtf?	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < m && col < n){
		float dotProd = 0.0;
		for(int i = 0; i < k; i++){
			dotProd += A[row * k + i] * B[i * n + col];
		}
		C[row * n + col] = alpha * dotProd;
	}

}

__global__ void gemm_shared(size_t m, size_t n, size_t k, size_t tile_width, const float *A, const float *B, float *C, const float alpha){
	int ty = threadIdx.y;
	int tx = threadIdx.x;

	int row = ty + blockIdx.y * blockDim.y;
	int col = tx + blockIdx.x * blockDim.x;

	__shared__ float A_SHARED[SHARED_SIZE];
	__shared__ float B_SHARED[SHARED_SIZE];
	
	float Cvalue = 0.f;
	for(int t = 0; t < k / tile_width; t++){
		// load into shared
		A_SHARED[ty * tile_width + tx] = A[(k * row + t * tile_width + tx)];
		B_SHARED[ty * tile_width + tx] = B[(t * tile_width + ty) * n + col];

		// sync
		__syncthreads();
		
			
		for(int i = 0; i < tile_width; i++){
			Cvalue += A_SHARED[tile_width * ty + i] * B_SHARED[i * tile_width + tx];
		}
		__syncthreads();
	}
	C[row * n + col] = Cvalue * alpha;
	
}

void fill_matrix(float *a, int m, int n){
	for(int row = 0; row < m; row++){
		for(int col = 0; col < n; col++){
			a[row * n + col] = 2.0;
		}
	}
}

void check_errors(int m, int n, int k, float alpha,
		  float *a, float *b, float *c){
	float *temp;
	temp = (float *)malloc(m * n * sizeof(float));	
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			temp[i * n + j] = 0.0;
			for(int off = 0; off < k; off++){
				temp[i * n + j]+= alpha * a[i * k + off] * b[off * n + j];
			}
		}
	}

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			assert(temp[i * n + j] == c[i * n + j]);
		}
	}
	printf("Succesfully completed BatChest!\n");
	free(temp);
}

int main(){
	size_t m, k, n;
	m = 2048;
	k = 2048;
	n = 2048;
	// A = m * k	
	float *ha, *da;
	size_t a_size = m * k * sizeof(float);

	ha = (float *)malloc(a_size);
	cudaMalloc(&da, a_size);
	
	// B = k * n
	float *hb, *db;
	size_t b_size = k * n * sizeof(float);

	hb = (float *)malloc(b_size);
	cudaMalloc(&db, b_size);

	// C = m * n
	float *hc, *dc;
	size_t c_size = m * n * sizeof(float);

	float alpha = 2.0;

	hc = (float *)malloc(c_size);
	cudaMalloc(&dc, c_size);

	// fill matrices
	fill_matrix(ha, m, k);
	fill_matrix(hb, k, n);

	cudaMemcpy(da, ha, a_size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, b_size, cudaMemcpyHostToDevice);

	dim3 threads(16,16);
	int blocksX = (n + threads.x - 1) / threads.x;
	int blocksY = (m + threads.y - 1) / threads.y;
	dim3 gridSize(blocksX, blocksY);
	dim3 gridSize_m(blocksY, blocksX);
	
	size_t tile_width = 16;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	gemm_v0<<<gridSize, threads>>>(m, n, k, da, db, dc, alpha); // ~ 9.8 ms
	//gemm_v1_global<<<gridSize_m, threads>>>(m, n, k, da, db, dc, alpha);
	//gemm_shared<<<gridSize, threads>>>(m, n, k, tile_width, da, db, dc, alpha);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Time in kernel: %f on: %zu elements\n.", ms, m * n);
	
	cudaMemcpy(hc, dc, c_size, cudaMemcpyDeviceToHost);
	
	check_errors(m, n, k, alpha, ha, hb, hc);
	return 0;
}
