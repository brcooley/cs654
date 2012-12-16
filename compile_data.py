import re
import json

def stats(raw):
	# print("Working on " + str(raw))
	return (min(raw), sum(raw)/len(raw), max(raw), sd(raw))

def sd(raw):
	mean = sum(raw)/len(raw)
	return (sum((x - mean) ** 2 for x in raw)/len(raw)) ** (.5)

def new(cur, old):
	return cur[0] != old[0] or cur[1] != old[1]

def main():

	for raw_data in ['mergesort/merge_omp_results.out','mergesort/merge_mpi_results.out','mergesort/merge_cuda_results.out','fft/fft_omp_results.out','fft/fft_mpi_results.out','fft/fft_cuda_results.out']:
		
		data = {}
		with open(raw_data, 'r') as f:

			v = 0
			k = 0
			last = (0, 0)
			f.readline()

			for line in f:
				try:
					v = float(line)
				except ValueError:
					k = tuple(re.findall(r'[0-9]+', line))[:-1]
					if new(k, last):
						if last != (0, 0):
							data[last].append(v)
						data[k] = []
						last = k
					else:
						data[k].append(v)
			data[last].append(v)

		for key in data.keys():
			# print('{}: {}'.format(key, data[key]))
			data[key] = stats(data[key])
			print('{}: {}'.format(key, data[key]))

		data_wrapped = { 'problem': raw_data.rsplit('_'), 'data': data }

		with open('final_results.dat', 'w+') as f_out: 
			print(json.dumps(data_wrapped), file=f_out)


if __name__ == '__main__':
	main()
