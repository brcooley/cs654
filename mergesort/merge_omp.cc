#include <omp.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
using namespace std;

vector<int> merge(const vector<int>& left, const vector<int>& right) {
	vector<int> result;
	unsigned left_it = 0, right_it = 0;

	while(left_it < left.size() && right_it < right.size())	{
		if(left[left_it] < right[right_it]) {
			result.push_back(left[left_it]);
			left_it++;
		}
		else {
			result.push_back(right[right_it]);
			right_it++;
		}
	}

	// Push the remaining data from both vectors onto the resultant
	while(left_it < left.size()) {
		result.push_back(left[left_it]);
		left_it++;
	}

	while(right_it < right.size()) {
		result.push_back(right[right_it]);
		right_it++;
	}

	return result;
}

vector<int> mergesort(vector<int>& vec, int threads) {
	// Termination condition: List is completely sorted if it
	// only contains a single element.
	if(vec.size() == 1)	{
		return vec;
	}

	// Determine the location of the middle element in the vector
	std::vector<int>::iterator middle = vec.begin() + (vec.size() / 2);

	vector<int> left(vec.begin(), middle);
	vector<int> right(middle, vec.end());

	// Perform a merge sort on the two smaller vectors
	if (threads > 1) {
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				left = mergesort(left, threads/2);
			}
			#pragma omp section
			{
				right = mergesort(right, threads - threads/2);
			}
	  	}
	}
	else {
	  left = mergesort(left, 1);
	  right = mergesort(right, 1);
	}

	return merge(left, right);
}

double getTime() {
	timeval thetime;
	gettimeofday( &thetime, 0 );
	return thetime.tv_sec + thetime.tv_usec / 1000000.0;
}

int main(int argc, char** argv) {
	if (argc != 3) { printf("usage: ./merge_omp ARRAY_SIZE NUM_THREADS\n"); exit(1); }

	int arraySize, numThreads;
	double totalTime, startTime;

	totalTime = 0;
	arraySize = atoi(argv[1]);
	numThreads = atoi(argv[2]);

	vector<int> v(arraySize);
	for (int i = 0; i < arraySize; ++i) {
		v[i] = ((int) random() / 100000.0);
	}

	omp_set_num_threads(numThreads);
	startTime = getTime();
	v = mergesort(v, numThreads);
	totalTime = getTime() - startTime;

	//for (long i=0; i<1000000; ++i)
	//  cout << v[i] << "\n";
	// cout << t2-t1<< "\n";
	
	printf("%.5f\n", totalTime);
	return 0;
}
