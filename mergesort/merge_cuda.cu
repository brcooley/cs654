/*
 *	CS420 Homework 2 - Brett Cooley (brcooley@email.wm.edu)
 *
 *	Changed sumseq program to implement mergesort on a given subset of an array.  Uses
 *		pthread_barrier_t to make sure all threads are in sync, and after each barrier
 *		1/2 of the threads exit, the other half merging their data and the date of the
 *		closest thread to them which exited.  Barriers are all initialized prior to
 *		any thread creation, and just used in order.
 *
 *	Threads/Array size 		1024 			1048576			8388608			16777216
 *		1 					.000/.000/.000	.313/.313/.312	2.87/2.87/2.87	 5.97/ 5.97/ 5.97
 *		2 					.000/.000/.000	.334/.334/.334	3.04/3.04/3.04	 6.32/ 6.33/ 6.33
 *		4 					.000/.000/.000	.352/.353/.354	3.42/3.42/3.42	 7.12/ 7.12/ 7.12
 *		8					.001/.001/.001	.423/.452/.490	4.24/4.25/4.29	 8.85/ 8.87/ 8.90
 *		16					.001/.001/.001	.459/.494/.523	5.02/5.05/5.10	10.47/10.50/10.54
 *		32					.002/.002/.002	.506/.541/.635	5.75/5.80/5.90	12.02/12.07/12.15
 *		64					.003/.003/.003	.571/.598/.622	6.36/6.43/6.53	13.49/13.54/13.59
 *		128					.006/.006/.006	.575/.626/.677	6.89/6.96/7.03	14.77/14.82/14.87
 *		256					.011/.011/.012	.619/.650/.733	7.28/7.56/7.75	15.84/15.93/16.06
 *		512					.023/.024/.025	.665/.704/.749	8.03/8.09/8.12	17.12/17.35/17.61
 */

#include <cuda.h>
#include <math.h>
#include <fenv.h>
#include <stdio.h>
#include <sys/time.h>

__global__
void mergesort(int *a);

double getTime();
void handleErrors(cudaError_t);


/* Prototypes. */
void *threadMain(void *arg);
int merge(int,long,long,long);

int main(int argc, char *argv[]) {
	if (argc != 3) { printf("usage: ./merge_cuda DEVICE ARRAY_SIZE NUM_THREADS\n"); exit(1); }

	int *a, *dev_a; //, *dev_b;
	double totalTime, startTime;

	totalTime = 0;
	int deviceNum = atoi(argv[1]);
	int arraySize = atoi(argv[2]);
	int numThreads = atoi(argv[3]);

	char x = ((cudaSetDevice(deviceNum))== cudaSuccess)? 'Y' : 'N';
	
	/* Initialize the sequence. */
	srand(654);
	a = (int *) malloc(sizeof(int) * arraySize);
	for(int i = 0; i < arraySize; i++) {
		a[i] = ((int) random() / 100000.0);
	}

	cudaMalloc((void **)&dev_a, sizeof(int) * arraySize);
	// cudaMalloc((void **)&dev_b, sizeof(int) * arraySize);
	cudaMemcpy(dev_a, &a, sizeof(int) * arraySize, cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_b, &b, sizeof(int) * NUM_VALUES, cudaMemcpyHostToDevice);

	// numThreads can range from 1..7 => {1,4,16,64,256,512,1024}
	int bH, bW, t;
	if (numThreads <= 2) {
		bH = (int) pow(4, numThreads-1);
		bw = 1;
	} else if (numThreads <= 4) {
		bH = bW = 4*(numThreads-2);
	} else {
		bH = bW = 16;
		t = ((int) pow(2, numThreads - 5);
	}

	dim3 blocks(bH, bW, 1);
	dim3 threads(t, 1, 1);

	startTime = getTime();
	mergesort<<<blocks,threads>>>(dev_a);
	cudaThreadSynchronize();
	totalTime += getTime() - startTime;
	handleErrors(cudaGetLastError());

	cudaFree(dev_a);
	printf("%.5f\n", totalTime);

	/*for(int i=0; i<arraySize; i++)
		printf("%f\n", a[outputArray][i]);
	printf("\n");*/

	return 0;
}


__global__
void mergesort(int *a) {
	int blockId = blockIdx.x * gridDim.y + blockIdx.y;
	int threadId = threadIdx.x * blockDim.y + threadIdx.y;
	int id = blockId * (blockDim.x * blockDim.y) + threadId;

	// int start =  * id;
}


void *threadMain(void *arg) {
	long start = baseSize * threadId;
	long subSize = baseSize;
	int target = 1;

	for (int j=0; j <= numRounds; j++) {
		for (int width=1; width < subSize; width *= 2) {
			for (int i=start; i<(start + subSize); i += 2 * width) {
				merge(target, i, i+width, i+(2*width));
			}
			target = 1 - target;
		}

		pthread_barrier_wait(&sync[j]);
		if (threadId == 0 || threadId % ((long) pow(2,j+1)) == 0) { subSize *= 2; }
		else { return (void *) 0; }
	}
	if (threadId == 0) { outputArray = 1-target; }
}


int merge(int target, long left, long right, long end) {
	long lEnd = right;
	for (int i=left; i<end; i++) {
		if (left < lEnd && (right >= end || values[1-target][left] <= values[1-target][right])) {
			values[target][i] = values[1-target][left];
			left++;
		} else {
			values[target][i] = values[1-target][right];
			right++;
		}
	}
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