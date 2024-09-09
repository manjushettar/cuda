#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include <math.h>

#define N 1024
#define TOLERANCE 1e-1

void fill_vector(float *a){
	for(int i = 0; i < N; i++){
		a[i] = (float)(rand() % 100);
	}
}

void verify_result(float *a, float *b, float *c, float factor){
	for(int i = 0; i < N; i++){
		float expected = factor * a[i] + b[i];
		assert(fabs(expected - c[i]) < TOLERANCE);
	}
}

int main(){
	size_t size = N * sizeof(float);
	float *ha, *hb, *hc;
	float *da, *db;

	ha = (float *) malloc(size);
	hb = (float *) malloc(size);
	hc = (float *) malloc(size);
	
	if(!ha || !hb || !hc){
		fprintf(stderr, "Failed to allocate host mem.");
		return -1;
	}

	cudaMalloc(&da, size);
	cudaMalloc(&db, size);
	
	fill_vector(ha);
	fill_vector(hb);
	
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);
	

	cublasSetVector(N, sizeof(float), ha, 1, da, 1);
	cublasSetVector(N, sizeof(float), hb, 1, db, 1);

	const float scale = 2.0f;
	status = cublasSaxpy(handle, N, &scale, da, 1, db, 1);
	// single precision A*X + Y
	
	cublasGetVector(N, sizeof(float), db, 1, hc, 1);

	verify_result(ha, hb, hc, scale);

	cublasDestroy(handle);
	cudaFree(da);
	cudaFree(db);
	free(ha);
	free(hb);	


	return 0;
}
