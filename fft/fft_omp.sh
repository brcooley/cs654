cd ~/Proj/cs654/fft
# g++ -Wall -o merge_omp -fopenmp merge_omp.cc && echo "Compilation successful"

date +%Y%m%d:%T >> fft_omp_results.out

for i in 4096 8192 16384; do
	for j in 1 2 4 8 16; do
		for k in 1 2 3 4 5 6 7 8 9 10; do
			echo "$j threads on matrix size $i [$k]: " >> fft_omp_results.out
			./fft_omp $i $j >> fft_omp_results.out
		done
	done
done
