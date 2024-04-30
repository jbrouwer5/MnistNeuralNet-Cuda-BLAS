Instructions to Locate and Run Code

CPU Native 
    - cpuNoBLAS.cpp
    - g++ -O3 cpuNoBLAS.cpp -o noblas
    - ./noblas 1 800 50 200 .1

CPU BLAS
    - module load intel
    - module load mkl
    - icpc cpuBLAS.cpp -o cpublas -m64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
    - ./cpublas 1 800 50 200 .1

GPU Native
    - module load cuda
    - module load gcc
    - nvcc -O3 -arch=sm_80 cuNoBLAS.cu -o noblas
    - ./noblas 1 800 50 200 .1

GPU BLAS
    - module load cuda
    - module load gcc
    - nvcc cuBLAS.cu -O3 -o cublas -lcublas
    - ./cublas 1 800 50 200 .1
