cd ~/Proj/cs654/mergesort
# g++ -Wall -o merge_omp -fopenmp merge_omp.cc && echo "Compilation successful"

date +%Y%m%d:%T >> merge_omp_results.out

for i in 1048576 16777216 536870912; do
	for j in 1 2 4 8 16; do
		for k in 1 2 3 4 5 6 7 8 9 10; do
			echo "$j threads on $i ints[$k]: " >> merge_omp_results.out
			./merge_omp $i $j >> merge_omp_results.out
		done
	done
done
