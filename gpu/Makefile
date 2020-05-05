all: gpu
gpu:
	nvcc -O3 main.cu -o md5_gpu -use_fast_math
clean:
	rm md5_gpu *.out *.err
