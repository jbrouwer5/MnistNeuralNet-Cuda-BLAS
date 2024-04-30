#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <cstdint>
#include <chrono> 
#include<cuda.h>
#include<curand.h>
#include<curand_kernel.h>
#include <cublas_v2.h>
#include "cuBlasUtility.cu"

__global__ void calcCost(int* d_order, int* d_labels, int outputSize, float* d_cost, int b, int nb, 
                         float* d_activationsMatrix, int nl, int i){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int c = 0; c < outputSize; c++){ 
        atomicAdd(d_cost, d_labels[d_order[b*nb+tid]*outputSize+c] * (-1.0*log(d_activationsMatrix[tid*outputSize+c])));
    }
}

__global__ void predict(int outputSize, float* activationsMatrix, int* testLabels, int* numCorrect, int i){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float maxVal = 0; 
    int maxIndex = 0; 
    for (int j = 0; j < outputSize; j++){
        if (activationsMatrix[tid*outputSize+j] > maxVal) {
            maxVal = activationsMatrix[tid*outputSize+j]; 
            maxIndex = j; 
        }
    }
    
    if (testLabels[(i+tid)*outputSize+maxIndex] == 1){
        atomicAdd(numCorrect, 1); 
    }
}
__global__ void flatRelu(float* results, float* inputs, int length, int numImages){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    results[tid] = max(inputs[tid], 0.0); 
}

__global__ void flatSoftmax(float* results, float* inputs, int length, int numImages){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float maxInput = inputs[tid*length];
    for (int i = 1; i < length; i++){
        if (inputs[tid*length+i] > maxInput) {
            maxInput = inputs[tid*length+i];
        }
    }

    float sum = 0;
    for (int j = 0; j < length; j++){
        sum += std::exp(inputs[tid*length+j] - maxInput); 
    }

    for (int i = 0; i < length; i++){
        results[tid*length+i] = std::exp(inputs[tid*length+i] - maxInput) / sum; 
    }
}

__global__ void flatMatrixMask(float* matrixA, float* matrixB) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    matrixA[tid] *= (matrixB[tid] > 0);
}

int main(int argc, char *argv[]) {
    if (argc != 6 && argc != 7) {
        printf("Usage: %s <nl> <nh> <ne> <nb> <alpha> (<ni>)\n", argv[0]);
        return 1;
    }

    // Parse Arguments ---------------------------------------------------------
    int nl = atoi(argv[1]); // Number of hidden layers
    int nh = atoi(argv[2]); // Number of activations in hidden layers
    int ne = atoi(argv[3]); // Number of epochs
    int nb = atoi(argv[4]); // Batch size
    float alpha = atof(argv[5]); // Learning rate
    int ni = nb; 
    if (argc == 7) {
        ni = atoi(argv[6]);
    }

    // Dimensions
    int inputSize = 784; // MNIST images are 28x28
    int outputSize = 10; // 10 classes for MNIST digits
    uint32_t imageCount = 50000, testImageCount = 10000, validationCount = 10000, totalCount = 60000; 

    // Get mnist data ----------------------------------------------------------
    // load mnist train images 
    float** mnistImages = readMnistImages("../train-images-idx3-ubyte", totalCount);
    
    
    // // load validation images
    // float** d_MnistValidationImages; 
    // cudaMalloc(&d_MnistValidationImages, validationCount * sizeof(float*));
    // for (int i = imageCount; i < totalCount; ++i) {
    //     float* deviceImage;
    //     cudaMalloc(&deviceImage, inputSize * sizeof(float)); 
    //     cudaMemcpy(deviceImage, mnistImages[i], inputSize * sizeof(float), cudaMemcpyHostToDevice); 
    //     cudaMemcpy(&d_MnistImages[i], &deviceImage, sizeof(float*), cudaMemcpyHostToDevice); 
    // }

    // load mnist test images 
    float** mnistTestImages = readMnistImages("../t10k-images-idx3-ubyte", testImageCount);
    float** d_mnistTestImages; 
    cudaMalloc(&d_mnistTestImages, testImageCount * sizeof(float*));
    for (int i = 0; i < testImageCount; ++i) {
        float* deviceImage;
        cudaMalloc(&deviceImage, inputSize * sizeof(float)); 
        cudaMemcpy(deviceImage, mnistTestImages[i], inputSize * sizeof(float), cudaMemcpyHostToDevice); 
        cudaMemcpy(&d_mnistTestImages[i], &deviceImage, sizeof(float*), cudaMemcpyHostToDevice); 
    }

    // load mnist train labels 
    int* labels = encodeHotLabelsFlat("../train-labels-idx1-ubyte", imageCount, outputSize); 
    float* labelsBatch = (float*)malloc(ni * outputSize * sizeof(float));
    float* d_labelsBatch; 
    cudaMalloc(&d_labelsBatch, ni * outputSize * sizeof(float));
    int* d_labels; 
    cudaMalloc(&d_labels, imageCount * outputSize * sizeof(int));
    cudaMemcpy(d_labels, labels, imageCount * outputSize * sizeof(int), cudaMemcpyHostToDevice); 
    // // load mnist validation labels
    // int** d_validationLabels; 
    // cudaMalloc(&d_validationLabels, validationCount * sizeof(int*));
    // for (int i = imageCount; i < imageCount + validationCount; ++i) {
    //     int* deviceImage;
    //     cudaMalloc(&deviceImage, outputSize * sizeof(int)); 
    //     cudaMemcpy(deviceImage, labels[i], outputSize * sizeof(int), cudaMemcpyHostToDevice); 
    //     cudaMemcpy(&d_labels[i], &deviceImage, sizeof(int*), cudaMemcpyHostToDevice); 
    // }

    // load mnist test labels 
    int* testLabels = encodeHotLabelsFlat("../t10k-labels-idx1-ubyte", testImageCount, outputSize); 
    int* d_testLabels; 
    cudaMalloc(&d_testLabels, testImageCount * outputSize * sizeof(int));
    cudaMemcpy(d_testLabels, testLabels, testImageCount * outputSize * sizeof(int), cudaMemcpyHostToDevice); 

    float **activationsMatrix = initActivationsMatrixFlat(nl, inputSize, outputSize, nh, ni);
    float **d_activationsMatrix = d_initActivationsMatrix(nl, inputSize, outputSize, nh, ni);
    float **d_weights = d_initWeights(nl, inputSize, outputSize, nh); 
    float **d_deltasMatrix = d_initDeltasMatrix(nl, outputSize, nh, ni);
    float **d_scoresMatrix = d_initScoresMatrix(nl, outputSize, nh, ni);
    float **d_biases = d_initBiases(nl, outputSize, nh); 
    
    int* order = initOrder(imageCount); 
    int* d_order;
    cudaMalloc((void**)&d_order, imageCount * sizeof(int));
    float **d_sumD = d_initDeltaSum(nl, outputSize, nh); 
    float **d_sumAD = d_initActivationDeltaSum(nl, inputSize, outputSize, nh); 
    float *cost = (float*)malloc(sizeof(float)); 
    float *d_cost; 
    cudaMalloc((void**)&d_cost, sizeof(float));
    
    // Initialize cuBLAS
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // training loop -----------------------------------------------------------
    for (int e = 0; e < ne; e++) {
        auto start = std::chrono::steady_clock::now();
    
        // shuffle the order of images for this epoch
        shuffle(order, imageCount); 
        cudaMemcpy(d_order, order, imageCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaError_t error3 = cudaGetLastError();
            if(error3 != cudaSuccess) {
                std::cout << "CUDA error3: " << cudaGetErrorString(error3) << std::endl;
                exit(-1);
            }
        *cost = 0; 
        cudaMemcpy(d_cost, cost, sizeof(float), cudaMemcpyHostToDevice);
        
        // Iterate Batches -----------------------------------------------------
        for (int b = 0; b < std::ceil(imageCount/float(nb)); b++) {
            // printWeights<<<1,1>>>(d_weights[0]); 
            // Reset weight errors
            for (int i = 0; i < nl + 1; i++){
                int rows = (i == 0) ? nh : ((i == nl) ? outputSize : nh);
                int cols = (i == 0) ? inputSize : nh;
                cudaMemset(d_sumAD[i], 0, rows * cols * sizeof(float));
            }
            
            // Reset bias error
            for (int i = 0; i < nl + 1; i++){
                int rows = (i == nl) ? outputSize : nh;
                cudaMemset(d_sumD[i], 0, rows * sizeof(float));
            }
            
            // Check if batchs don't fit into number of samples
            int currNb = nb; 
            if (b == (imageCount / nb) && imageCount % nb != 0){
                currNb = imageCount % nb; 
            }
            
            // Breaks batch into chunks of concurrent images -------------------
            int chunkSize = std::min(currNb, ni); 
            for (int i = 0; i < currNb; i += chunkSize) {
                int actualChunkSize = std::min(chunkSize, currNb - i);

                // Put inputs into first activation layer
                for (int k = 0; k < actualChunkSize; k++){
                    for (int c = 0; c < inputSize; c++){
                        activationsMatrix[0][k*inputSize+c] = mnistImages[order[b*nb+k]][c];
                    }
                }
                
                
                cudaMemcpy(d_activationsMatrix[0], activationsMatrix[0], inputSize * actualChunkSize * sizeof(float), cudaMemcpyHostToDevice);
                
                
                if (b == 0){
                    // printDeltas<<<1,1>>>(d_activationsMatrix[0]); 
                }
                // Forward Propogation -----------------------------------------
                for (int j = 0; j < nl; j++) {
                    int rows = nh;
                    int cols = (j == 0) ? inputSize : nh;
                    float scalar = 1.0; 
                    float beta = 0.0; 
                    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, actualChunkSize, 
                                rows, cols, &scalar, d_activationsMatrix[j], actualChunkSize, 
                                d_weights[j], rows, &beta, 
                                d_scoresMatrix[j + 1], actualChunkSize);
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "cuBLAS initialization failed1\n";
                        return EXIT_FAILURE;
                    }

                    
                    for (int s = 0; s < actualChunkSize; s++) { 
                        stat = cublasSaxpy(handle, rows, &scalar, d_biases[j], 1, d_scoresMatrix[j+1] + s * rows, 1);
                        
                        if (stat != CUBLAS_STATUS_SUCCESS) {
                            std::cerr << "cuBLAS initialization failed2\n";
                            return EXIT_FAILURE;
                        }
                    }
                    
                    flatRelu<<<rows, actualChunkSize>>>(d_activationsMatrix[j+1], d_scoresMatrix[j+1], rows, actualChunkSize);
                    
                }
                
                float scalar = 1.0; 
                float beta = 0.0; 
                stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, actualChunkSize, 
                    outputSize, nh, &scalar, d_activationsMatrix[nl], actualChunkSize, 
                    d_weights[nl], outputSize, &beta, 
                    d_scoresMatrix[nl + 1], actualChunkSize);
                
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "cuBLAS initialization failed4\n";
                    return EXIT_FAILURE;
                }
                for (int s = 0; s < actualChunkSize; s++) { 
                    stat = cublasSaxpy(handle, outputSize, &scalar, d_biases[nl], 1, d_scoresMatrix[nl+1] + s * outputSize, 1);
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "cuBLAS initialization failed3\n";
                        return EXIT_FAILURE;
                    }
                }
                
                
                flatSoftmax<<<actualChunkSize, 1>>>(d_activationsMatrix[nl+1], d_scoresMatrix[nl+1], outputSize, actualChunkSize); 
                
                // printDeltas<<<1,1>>>(d_activationsMatrix[nl+1]); 
                calcCost<<<actualChunkSize, 1>>>(d_order, d_labels, outputSize, d_cost, b, nb, 
                                                 d_activationsMatrix[nl+1], nl, i); 
                                         
                // Backwards Propogation ---------------------------------------
                for (int k = 0; k < actualChunkSize; k++){
                    for (int m = 0; m < outputSize; m++){
                        labelsBatch[k*outputSize+m] = labels[order[b*nb+k]*outputSize+m]; 
                    }
                }
                
                cudaMemcpy(d_labelsBatch, labelsBatch, outputSize * actualChunkSize * sizeof(float), cudaMemcpyHostToDevice);
                
                if (b == 0){
                    // printLabels<<<1,1>>>(d_labelsBatch); 
                }
                scalar = -1.0; 
                for (int s = 0; s < actualChunkSize; s++) {
                    stat = cublasScopy(handle, outputSize, 
                        d_activationsMatrix[nl+1]+s*outputSize, 1, 
                        d_deltasMatrix[nl]+s*outputSize, 1);
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "cuBLAS initialization failed\n";
                        return EXIT_FAILURE;
                    }
                    
                    stat = cublasSaxpy(handle, outputSize, &scalar, 
                        d_labelsBatch+s*outputSize, 1, 
                        d_deltasMatrix[nl]+s*outputSize, 1);
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "cuBLAS initialization failed\n";
                        return EXIT_FAILURE;
                    }
                }
                // printDeltas<<<1,1>>>(d_deltasMatrix[nl]); 
               
                
                scalar = 1.0;
                beta = 1.0;

                // calculate error
                stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                            outputSize, nh, actualChunkSize, 
                            &scalar, 
                            d_deltasMatrix[nl], actualChunkSize, 
                            d_activationsMatrix[nl], actualChunkSize, 
                            &beta, 
                            d_sumAD[nl], outputSize);
                             
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "7cuBLAS initialization failed\n";
                    return EXIT_FAILURE;
                }
                for (int i = 0; i < actualChunkSize; i++) {
                    stat = cublasSaxpy(handle, outputSize, &scalar, 
                                d_deltasMatrix[nl] + i * outputSize, 1, 
                                d_sumD[nl], 1);
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "8cuBLAS initialization failed\n";
                        return EXIT_FAILURE;
                    }
                }

                scalar = 1.0;
                beta = 0.0;
                // stopped here
                for (int l = nl; l > 0; l--){
                    int rows = (l == nl) ? outputSize : nh;
                    int cols = nh;
                    int lowCols = (l == 1) ? inputSize : nh;

                    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                actualChunkSize, cols, rows, 
                                &scalar,
                                d_deltasMatrix[l], actualChunkSize, 
                                d_weights[l], cols, 
                                &beta,
                                d_deltasMatrix[l-1], actualChunkSize);
                                
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "9cuBLAS initialization failed\n";
                        return EXIT_FAILURE;
                    }
                    
                    flatMatrixMask<<<actualChunkSize, cols>>>(d_deltasMatrix[l-1], d_scoresMatrix[l]); 
                    
                    // Perform the accumulation in a loop, similar to the original C code
                    for (int i = 0; i < actualChunkSize; i++) {
                        stat = cublasSaxpy(handle, cols, &scalar, 
                                    d_deltasMatrix[l] + i * cols, 1, 
                                    d_sumD[l], 1);
                        
                        if (stat != CUBLAS_STATUS_SUCCESS) {
                            std::cerr << "10cuBLAS initialization failed\n";
                            return EXIT_FAILURE;
                        }
                    }

                    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                                cols, lowCols, actualChunkSize, 
                                &scalar, 
                                d_deltasMatrix[l-1], actualChunkSize, 
                                d_activationsMatrix[l-1], actualChunkSize,        
                                &scalar, 
                                d_sumAD[l-1], cols); 
                    
                    if (stat != CUBLAS_STATUS_SUCCESS) {
                        std::cerr << "11cuBLAS initialization failed\n";
                        return EXIT_FAILURE;
                    }
                }
            }
            float scaleFactor = -(alpha / currNb); 
            // Update Weights Based on Errors ----------------------------------
            for (int l = nl; l >= 0; l--){
                int rows = (l == 0) ? nh : ((l == nl) ? outputSize : nh);
                int cols = (l == 0) ? inputSize : nh;
                stat = cublasSaxpy(handle, rows*cols, &scaleFactor, 
                            d_sumAD[l], 1, 
                            d_weights[l], 1);
                
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "12cuBLAS initialization failed\n";
                    return EXIT_FAILURE;
                }
                stat = cublasSaxpy(handle, rows, &scaleFactor, 
                    d_sumD[l], 1, 
                    d_biases[l], 1);
                
                if (stat != CUBLAS_STATUS_SUCCESS) {
                    std::cerr << "13cuBLAS initialization failed\n";
                    return EXIT_FAILURE;
                }
            }
        }
        
        cudaMemcpy(cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
        
        
        std::cout << "Loss for epoch " << e << " is " << *cost / (50000) << std::endl; 
        std::cout << "Took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count() << " milliseconds" << std::endl;

        // Predict the labels of test images -----------------------------------
        auto start2 = std::chrono::steady_clock::now();
        int *numCorrect = (int*)malloc(sizeof(int)); 
        *numCorrect = 0; 
        int *d_numCorrect; 
        cudaMalloc(&d_numCorrect, sizeof(int));
        cudaMemcpy(d_numCorrect, numCorrect, sizeof(int), cudaMemcpyHostToDevice);
        
        for (int i = 0; i < testImageCount; i += ni) {
            int actualChunkSize = std::min(ni, int(testImageCount) - i);
            
            // Put inputs into first activation layer
            for (int k = 0; k < actualChunkSize; k++){
                for (int c = 0; c < inputSize; c++){
                    activationsMatrix[0][k*inputSize+c] = mnistTestImages[i+k][c];
                }
            }
            cudaMemcpy(d_activationsMatrix[0], activationsMatrix[0], inputSize * actualChunkSize * sizeof(float), cudaMemcpyHostToDevice);
            
            // Forward Propogation -----------------------------------------
            for (int j = 0; j < nl; j++) {
                int rows = nh;
                int cols = (j == 0) ? inputSize : nh;
                float scalar = 1.0; 
                float beta = 0.0; 
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, actualChunkSize, 
                            rows, cols, &scalar, d_activationsMatrix[j], actualChunkSize, 
                            d_weights[j], rows, &beta, 
                            d_scoresMatrix[j + 1], actualChunkSize);
                // 
                for (int s = 0; s < actualChunkSize; s++) { 
                    cublasSaxpy(handle, rows, &scalar, d_biases[j], 1, d_scoresMatrix[j+1] + s * rows, 1);
                    // 
                }
                flatRelu<<<rows, actualChunkSize>>>(d_activationsMatrix[j+1], d_scoresMatrix[j+1], rows, actualChunkSize);
                // 
            }
            
            float scalar = 1.0; 
            float beta = 0.0; 
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, outputSize, 
                actualChunkSize, nh, &scalar, d_weights[nl], nh, 
                d_activationsMatrix[nl], nh, &beta, 
                d_scoresMatrix[nl + 1], outputSize);
                
            for (int s = 0; s < actualChunkSize; s++) { 
                cublasSaxpy(handle, outputSize, &scalar, d_biases[nl], 1, d_scoresMatrix[nl+1] + s * outputSize, 1);
                
            }
            
            flatSoftmax<<<actualChunkSize, 1>>>(d_activationsMatrix[nl+1], d_scoresMatrix[nl+1], outputSize, actualChunkSize); 
            
            
            predict<<<actualChunkSize, 1>>>(outputSize, d_activationsMatrix[nl+1], d_testLabels, d_numCorrect, i);
            
            cudaError_t error3 = cudaGetLastError();
            if(error3 != cudaSuccess) {
                std::cout << "CUDA error3: " << cudaGetErrorString(error3) << std::endl;
                exit(-1);
            }
            
        }
        cudaMemcpy(numCorrect, d_numCorrect, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Prediction Rate: " << float(*numCorrect) / testImageCount << std::endl;
        std::cout << "Prediction took " << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start2).count() << " milliseconds" << std::endl;
    }

    // Free Matrices -----------------------------------------------------------
    // freeMatrix(activationsMatrix, nl+2); 
    // freeMatrix(deltasMatrix, nl+1);
    // freeMatrix(scoresMatrix, nl+2); 
    // freeMatrix(weights, nl+1); 
    // free(sumAD); 
    // freeMatrix(mnistImages, imageCount); 
    // freeMatrix(mnistTestImages, testImageCount);
    // freeMatrix(labels, imageCount); 
    // freeMatrix(testLabels, testImageCount); 
    // freeMatrix(biases, nl+1);
    // freeMatrix(sumD, nl+1); 
    // free(order); 
    
    return 0;
}