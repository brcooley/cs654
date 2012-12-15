cd ~/Proj/cs654/fft
# g++ -Wall -o merge_omp -fopenmp merge_omp.cc && echo "Compilation successful"

date +%Y%m%d:%T >> fft_mpi_results.out

for i in 4096 8192 16384; do
	for j in 1 2 4; do
		for k in 1 2 3 4 5 6 7 8 9 10; do
			echo "$j processors on matrix size $i [$k]: " >> fft_mpi_results.out
			mpirun -n $j ./fft_mpi $i >> fft_mpi_results.out
		done
	done
	for j in 8 16; do
		for k in 1 2 3 4 5 6 7 8 9 10; do
			echo "$j processors on matrix size $i [$k]: " >> fft_mpi_results.out
			mpirun -n $j --host bg7,bg8,bg9,bg2 ./fft_mpi $i >> fft_mpi_results.out
		done
	done
done
