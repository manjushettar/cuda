#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

__global__ void forward(int batch_size, int neurons, int out_size, float *in, float* weights, float *biases, float *out){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < batch_size && col < out_size){
		float dotProd = 0.0f;
		for(int i = 0; i < neurons; i++){
			dotProd += in[row * neurons + i] * weights[out_size * i + col];
		}
		out[row * out_size + col] = dotProd + biases[col];
	}

}

__global__ void relu(float* in, float* out, int w, int h){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < h && col < w){
		float activation = in[row * w + col];
		if(activation < 0.0){
			out[row * w + col] = 0.0;
		}else{
			out[row * w + col] = activation;
		}
	}
}


__global__ void softmax(float *in, float *out, int w, int h){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < w && col < h){
		float max_val = in[row * w];
		for(int i = 1; i < w; i++){
			max_val = max(max_val, in[row * w + i];
		}
		float divisor = 0.0f;
		for(int i = 0; i < w; i++){
			divisor += exp(input[row * w + i] - max_val);
		}
		out[row * w + col] = exp(input[row * w + col] - max_val) / divisor;
	}
}

__global__ void loss(float *out, float *p, float *r, int w, int h){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < h){
		float loss = 0.f;
		for(int i = 0; i < h; i++){
			loss -= r[col * w + i] * log(max(1e-6, p[col * w + i]));
		}
		out[col] = loss;
	}
}

__global__ void init_weights(int w, int h, float *out){
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	if(row < h && col < h){
		curandState state;
		curand_init(42, row * w + col, 0, &state);
		out[row*w + col] = curand_normal(&state) * sqrtf(2.f/h);
	}
}
