#include <cuda.h>
#include <cufft.h>
#include <math.h>
#include <fenv.h>
#include <stdio.h>
#include <sys/time.h>

double getTime();
void handleErrors(cudaError_t);


int main(int argc, char *argv[]) {
	if (argc != 3) { printf("usage: ./fft_cuda DEVICE MATRIX_SIZE NUM_THREADS\n"); exit(1); }


	cufftHandle plan;
	cufftComplex *data, *dev_data;

	double totalTime, startTime;
	totalTime = 0;

	int deviceNum = atoi(argv[1]);
	int matrixSize = atoi(argv[2]);
	int numThreads = atoi(argv[3]);

	char x = ((cudaSetDevice(deviceNum))== cudaSuccess)? 'Y' : 'N';
	
	/* Initialize the sequence. */
	srand(654);
	data = (cufftComplex *) malloc(sizeof(cufftComplex) * matrixSize * matrixSize);
	int i, j;
	// double pdata=0;
	for (i = 0; i < matrixSize; ++i) {
		for (j = 0; j < matrixSize; ++j) {
			data[i*matrixSize+j].x = i;
			data[i*matrixSize+j].y = 0;
			// pdata += data[i*matrixSize+j][0] * data[i*matrixSize+j][0] + data[i*matrixSize+j][1] * data[i*matrixSize+j][1];
		}
	}

	/* Create a 2D FFT plan. */
	if (cufftPlan2d(&plan, matrixSize, matrixSize, CUFFT_C2C) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return 0;	
	}

	if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
		return 0;		
	}

	cudaMalloc((void **)&dev_data, sizeof(cufftComplex) * matrixSize * matrixSize);	
	cudaMemcpy(dev_data, &data, sizeof(cufftComplex) * matrixSize * matrixSize, cudaMemcpyHostToDevice);

	// numThreads can range from 1..7 => {1,4,16,64,256,512,1024}
	int bH, bW, t;
	if (numThreads <= 2) {
		bH = (int) pow(4, numThreads-1);
		bW = 1;
	} else if (numThreads <= 4) {
		bH = bW = 4*(numThreads-2);
	} else {
		bH = bW = 16;
		t = (int) pow(2, numThreads - 5);
	}

	dim3 blocks(bH, bW, 1);
	dim3 threads(t, 1, 1);

	startTime = getTime();
	cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	cudaThreadSynchronize();
	totalTime += getTime() - startTime;
	handleErrors(cudaGetLastError());

	cufftDestroy(plan);
	cudaFree(data);

	printf("%.5f\n", totalTime);

	return 0;
}


double getTime() {
	timeval thetime;
	gettimeofday( &thetime, 0 );
	return thetime.tv_sec + thetime.tv_usec / 1000000.0;
}


void handleErrors(cudaError_t err) {
	if (err != cudaSuccess) {
		printf("Kernel: %s\n",cudaGetErrorString(err));
	}
}
