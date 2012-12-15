cd ~/Proj/cs654/mergesort
# g++ -Wall -o merge_omp -fopenmp merge_omp.cc && echo "Compilation successful"

date +%Y%m%d:%T >> merge_mpi_results.out

for i in 1048576 16777216 536870912; do
	for j in 1 2 4; do
		for k in 1 2 3 4 5 6 7 8 9 10; do
			echo "$j threads on $i ints[$k]: " >> merge_mpi_results.out
			mpirun -n $j ./merge_mpi $i >> merge_mpi_results.out
		done
	done
	for j in 8 16; do
		 for k in 1 2 3 4 5 6 7 8 9 10; do
                        echo "$j threads on $i ints[$k]: " >> merge_mpi_results.out
                        mpirun -n $j --host bg7,bg8,bg9,bg2 ./merge_mpi $i >> merge_mpi_results.out
                done
	done
done
