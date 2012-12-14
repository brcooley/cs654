#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <fenv.h>
#include <sys/time.h>

void showVector(int *v, int n, int id);
int * merge(int *A, int asize, int *B, int bsize);
void swap(int *v, int i, int j);
void m_sort(int *A, int min, int max);
double getTime();

double startTime, stopTime;


void showVector(int *v, int n, int id) {
	int i;
	printf("%d: ",id);
	for(i = 0 ; i < n; i++) {
		printf("%d ",v[i]);
	}
	printf("\n");
}

int * merge(int *A, int asize, int *B, int bsize) {
	int ai, bi, ci, i;
	int *C;
	int csize = asize+bsize;

	ai = 0;
	bi = 0;
	ci = 0;

	/* printf("asize=%d bsize=%d\n", asize, bsize); */

	C = (int *) malloc(csize*sizeof(int));
	while ((ai < asize) && (bi < bsize)) {
		if (A[ai] <= B[bi]) {
			C[ci] = A[ai];
			ci++; ai++;
		} else {
			C[ci] = B[bi];
			ci++; bi++;
		}
	}

	if (ai >= asize) {
		for (i = ci; i < csize; i++, bi++) {
			C[i] = B[bi];
		}
	} else if (bi >= bsize) {
		for (i = ci; i < csize; i++, ai++) {
			C[i] = A[ai];
		}
	}
	for (i = 0; i < asize; i++) {
		A[i] = C[i];
	}
	for (i = 0; i < bsize; i++) {
		B[i] = C[asize+i];
	}

	/* showVector(C, csize, 0); */
	return C;
}

void swap(int *v, int i, int j) {
	int t;
	t = v[i];
	v[i] = v[j];
	v[j] = t;
}

void m_sort(int *A, int min, int max) {
	int *C;		/* dummy, just to fit the function */
	int mid = (min+max)/2;
	int lowerCount = mid - min + 1;
	int upperCount = max - mid;

	/* If the range consists of a single element, it's already sorted */
	if (max == min) {
		return;
	} else {
		/* Otherwise, sort the first half */
		m_sort(A, min, mid);
		/* Now sort the second half */
		m_sort(A, mid+1, max);
		/* Now merge the two halves */
		C = merge(A + min, lowerCount, A + mid + 1, upperCount);
	}
}


double getTime() {
	timeval thetime;
	gettimeofday( &thetime, 0 );
	return thetime.tv_sec + thetime.tv_usec / 1000000.0;
}


int main(int argc, char **argv) {
	int * data;
	int * chunk;
	int * other;
	int m;
	int id, p;
	int s = 0;
	int i;
	int step;
	MPI_Status status;

	if (argc != 2) { printf("usage: ./merge_mpi ARRAY_SIZE\n"); exit(1); }

	int arraySize = atoi(argv[1]);

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if (id == 0)	{
		int r;
		srandom(clock());
		s = arraySize / p;
		r = arraySize % p;
		data = (int *) malloc((arraySize + s - r) * sizeof(int));
		for(i = 0; i < arraySize; i++) {
			data[i] = random();
		}
		if (r != 0) {
			for (i = arraySize; i < arraySize + s - r; i++)
				data[i] = 0;
			s += 1;
		}

		MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
		chunk = (int *) malloc(s * sizeof(int));
		MPI_Scatter(data, s, MPI_INT, chunk, s, MPI_INT, 0, MPI_COMM_WORLD);
		m_sort(chunk, 0, s - 1);
		/* showVector(chunk, s, id); */
	}
	else {
		MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
		chunk = (int *) malloc(s * sizeof(int));
		MPI_Scatter(data, s, MPI_INT, chunk, s, MPI_INT, 0, MPI_COMM_WORLD);
		m_sort(chunk, 0, s-1);
		/* showVector(chunk, s, id); */
	}

	startTime = getTime();
	step = 1;
	while (step < p) {
		if (id % (2 * step) == 0) {
			if (id + step < p) {
				MPI_Recv(&m, 1, MPI_INT, id + step, 0, MPI_COMM_WORLD, &status);
				other = (int *) malloc(m * sizeof(int));
				MPI_Recv(other, m, MPI_INT, id + step, 0, MPI_COMM_WORLD, &status);
				chunk = merge(chunk, s, other, m);
				s += m;
			} 
		}
		else {
			int near = id - step;
			MPI_Send(&s, 1, MPI_INT, near, 0, MPI_COMM_WORLD);
			MPI_Send(chunk, s, MPI_INT, near, 0, MPI_COMM_WORLD);
			break;
		}
		step *= 2;
	}

	stopTime = getTime();

	if (id == 0) {
		printf("%.5f\n", (stopTime-startTime));
	}
	MPI_Finalize();
	return 0;
}
