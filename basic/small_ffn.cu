#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <cublas_v2.h>
#define N 64
#define F 1e-5

__global__ void weightMul(float *w, float *in, float factor, float *out){
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < N){
		out[index] = factor * in[index] + w[index];
	}
}

void fill_vector(float *in){
	for(int i = 0; i < N; i++){
		float temp = static_cast<float>(rand()) / static_cast <float> (RAND_MAX);
		in[i] = temp;		
	}
}

void print_vector(const char* name, float *vec){
	printf("%s: ", name);
	for(int i = 0; i < N; i++){
		printf("%f ", vec[i]);
	}
	printf("\n");
}

void check_errors(float *w, float *in, float *c, float factor){
	for(int i = 0; i < N; i++){
		float temp;
		temp = factor * in[i] + w[i];
		float error;
		error = fabs(temp - c[i]);
		printf("%f, temp: %f\n", c[i], temp);
		assert(error < F);
	}
}

int main(){
	size_t size = N * sizeof(float);
	float *ha, *hb, *hc, *dw, *din;

	ha = (float *)(malloc(size));
	hb = (float *)(malloc(size));
	hc = (float *)(malloc(size));

	cudaMalloc(&dw, size);
	cudaMalloc(&din, size);

	fill_vector(ha);
	fill_vector(hb);

	//print_vector("Input ha", ha);
	//print_vector("Input hb", hb);

	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	
	cudaMemcpy(dw, ha, size, cudaMemcpyHostToDevice);
	cudaMemcpy(din, hb, size, cudaMemcpyHostToDevice);
	
	float const factor = 2.0f;
	cublasSaxpy(handle, size,cublasHandle_t handle;
        cublasCreate_v2(&handle);

        cudaMemcpy(dw, ha, size, cudaMemcpyHostToDevice);
        cudaMemcpy(din, hb, size, cudaMemcpyHostToDevice);

        float *dout;
        cudaMalloc(&dout, size);

        float const factor = 2.0f;
        cublasSaxpy(handle, N, &factor, din, 1, dw, 1);
        cudaMemcpy(hc, dw, size, cudaMemcpyDeviceToHost);

        check_errors(ha, hb, hc, factor);
        printf("No errors. Batchest!\n"); &factor, din, 1, dw, 1);
	cudaMemcpy(hc, dw, size, cudaMemcpyDeviceToHost);
		
	check_errors(ha, hb, hc, factor);
	printf("No errors. Batchest!\n");
	
	cublasDestroy(handle);
	free(ha);
	free(hb);
	free(hc);
	cudaFree(din);
	cudaFree(dw);
	return 0;
}
