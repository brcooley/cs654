import re
import json

def stats(raw):
	return (min(raw), sum(raw)/len(raw), max(raw), sd(raw))

def sd(raw):
	mean = sum(raw)/len(raw)
	return (sum((x - mean) ** 2 for x in raw)/len(raw)) ** (.5)

def new(cur, old):
	return cur[0] != old[0] or cur[1] != old[1]

def main():

	for raw_data in ['merge_omp_results.out','merge_mpi_results.out','merge_cuda_results.out','fft_omp_results.out','fft_mpi_results.out','fft_cuda_results.out']:
		
		data = {}
		with open(raw_data, 'r') as f:

			v = 0
			k = 0
			last = (0, 0, 0)

			for line in f:
				try:
					v = float(line)
				except ValueError:
					k = tuple(re.findall(r'[0-9]+', line))[:-1]
					if new(k, last):
						data[k] = []
						last = k
					else:
						data[k].append(v)

		for key in data.keys():
			data[key] = stats(data[key])


		data_wrapped = { 'problem': raw_data.rsplit('_'), 'data': data }

		with open('final_results.dat', 'w+') as f_out: 
			print(json.dumps(data_wrapped), file=f_out)


if __name__ == '__main__':
	main()
