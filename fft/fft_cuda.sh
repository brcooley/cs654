cd ~/Proj/cs654/fft
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-4.1.28/lib:/usr/local/cuda-4.1.28/lib64"

date +%Y%m%d:%T >> fft_cuda_results.out

for i in 4096 8192 16384; do
	for k in 1 2 3 4 5; do
		echo "Matrix size $i [$k]: " >> fft_cuda_results.out 
		(./fft_cuda 0 $i >> fft_cuda_results.out)
		(./fft_cuda 1 $i >> fft_cuda_results.out)
	done
done
